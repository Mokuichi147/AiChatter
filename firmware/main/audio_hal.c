#include "audio_hal.h"

#include <string.h>

#include "config.h"
#include "driver/i2c_master.h"   /* 新I2C API (IDF 5.x) */
#include "driver/i2s_std.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

/* 簡易エネルギーベースVAD (ESP-SRは静的メモリ不足のため未使用) */

#define TAG "AUDIO_HAL"

/* 再生用リングバッファサイズ (約500ms分 @ 16kHz, 16bit) */
#define PLAYBACK_RINGBUF_SIZE   (AUDIO_CHUNK_BYTES * 16)
/* AECリファレンス用リングバッファサイズ (約250ms分) */
#define AEC_REF_RINGBUF_SIZE    (AUDIO_CHUNK_BYTES * 8)

static i2s_chan_handle_t s_tx_handle = NULL;
static i2s_chan_handle_t s_rx_handle = NULL;
static esp_codec_dev_handle_t s_codec_dev = NULL;
static bool s_codec_available = false;

/* 共有I2Cバスハンドル (pmic_lcd_power_on()で初期化, codec_init()で再利用) */
static i2c_master_bus_handle_t s_i2c_bus_handle = NULL;

/* 簡易VAD設定
 * ノイズフロア: rms²≈150〜300 (Philips I2S, PGA +24dB)
 * 発話時: rms²≈50,000以上 */
#define VAD_RMS_SQ_THRESHOLD    5000LL  /* 発話検出閾値 (rms²) */
#define VAD_SPEECH_FRAMES       3             /* 連続発話フレーム数で確定 */
#define VAD_SILENCE_FRAMES      15            /* 連続無音フレーム数で確定 */

/* リングバッファ */
static RingbufHandle_t s_aec_ref_buf  = NULL;
static RingbufHandle_t s_playback_buf = NULL;

static audio_rx_callback_t  s_rx_cb  = NULL;
static vad_event_callback_t s_vad_cb = NULL;

static volatile bool s_stop_playback = false;

/* --------------------------------------------------------
 * I2Sフルデュプレックス初期化
 * -------------------------------------------------------- */
static void i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(
        I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;

    /* TX と RX を同じ I2S ペリフェラルで同時生成 (フルデュプレックス) */
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &s_tx_handle, &s_rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_MCLK_GPIO,
            .bclk = I2S_BCLK_GPIO,
            .ws   = I2S_WS_GPIO,
            .dout = I2S_DOUT_GPIO,
            .din  = I2S_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    /* ES8311はモノラル出力を左チャンネルに配置する */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx_handle));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_handle));

    ESP_LOGI(TAG, "I2Sフルデュプレックス初期化完了 (%dkHz, 16bit, mono)",
             AUDIO_SAMPLE_RATE / 1000);
}

/* --------------------------------------------------------
 * PMIC (M5PM1, 0x6E) レジスタ書き込み
 * ※ ESP-IDF 5.x新I2C APIではi2c_master_transmit_receiveが
 *   M5PM1と互換性がなくタイムアウトする。書き込みのみ使用。
 *   M5Unifiedは旧I2C API(i2c_cmd_link)で読み込みに成功している。
 * -------------------------------------------------------- */
static esp_err_t pmic_write_reg(i2c_master_dev_handle_t dev,
                                 uint8_t reg, uint8_t val) {
    uint8_t buf[2] = { reg, val };
    esp_err_t ret = i2c_master_transmit(dev, buf, 2, pdMS_TO_TICKS(100));
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "PMIC reg0x%02X=0x%02X 書き込み失敗: %s",
                 reg, val, esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "PMIC reg0x%02X = 0x%02X 書き込みOK", reg, val);
    }
    return ret;
}

/* --------------------------------------------------------
 * PMIC (M5PM1, 0x6E) 初期化 & LCD/ES8311電源有効化
 * lcd_init() より前に app_main() から呼び出すこと。
 *
 * M5Unified初期化シーケンスから最終状態を特定し直接書き込む。
 * (I2C読み込み不可のためread-modify-writeは使えない)
 *
 * 各レジスタの値 (M5Unified解析結果):
 *   0x09 = 0x00 : I2Cアイドルスリープ無効化
 *   0x16 = 0x00 : GPIO0,2,3 → GPIO機能 (bit0,2,3=0)
 *   0x10 = 0x0C : GPIO2,3=出力(bit2,3=1), GPIO0=入力(bit0=0)
 *   0x13 = 0x00 : GPIO2,3 → プッシュプル (bit2,3=0)
 *   0x11 = 0x0C : GPIO2=HIGH(LCD電源), GPIO3=HIGH(AW8737/ES8311)
 * ※ 0x06 (5V出力) は触らない (デバイスリセットを引き起こすため)
 * -------------------------------------------------------- */
void pmic_lcd_power_on(void) {
    if (s_i2c_bus_handle != NULL) return;  /* 既に初期化済み */

    i2c_master_bus_config_t i2c_bus_cfg = {
        .clk_source        = I2C_CLK_SRC_DEFAULT,
        .i2c_port          = I2C_NUM_0,
        .sda_io_num        = I2C_SDA_GPIO,
        .scl_io_num        = I2C_SCL_GPIO,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t bus_err = i2c_new_master_bus(&i2c_bus_cfg, &s_i2c_bus_handle);
    if (bus_err != ESP_OK) {
        ESP_LOGE(TAG, "I2Cバス初期化失敗: %s", esp_err_to_name(bus_err));
        return;
    }

    i2c_device_config_t pmic_dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = PMIC_I2C_ADDR,
        .scl_speed_hz    = 100000,
    };
    i2c_master_dev_handle_t pmic_dev = NULL;
    if (i2c_master_bus_add_device(s_i2c_bus_handle, &pmic_dev_cfg, &pmic_dev)
            != ESP_OK) {
        ESP_LOGW(TAG, "PMIC (0x%02X) デバイス追加失敗", PMIC_I2C_ADDR);
        return;
    }
    ESP_LOGI(TAG, "PMIC: デバイス追加完了 (0x%02X)", PMIC_I2C_ADDR);

    /* I2Cアイドルスリープ無効化 */
    pmic_write_reg(pmic_dev, 0x09, 0x00);

    /* GPIO0,2,3をGPIO機能に (代替機能OFF) */
    pmic_write_reg(pmic_dev, 0x16, 0x00);

    /* GPIO2,3を出力モード、GPIO0を入力モード */
    pmic_write_reg(pmic_dev, 0x10, 0x0C);

    /* GPIO2,3をプッシュプルモード */
    pmic_write_reg(pmic_dev, 0x13, 0x00);

    /* GPIO2=HIGH(LCD電源ON), GPIO3=HIGH(AW8737/ES8311電源ON) */
    pmic_write_reg(pmic_dev, 0x11, 0x0C);

    /* ES8311は電源投入後にI2Cスレーブが安定するまで時間が必要 */
    vTaskDelay(pdMS_TO_TICKS(500));

    ESP_LOGI(TAG, "PMIC初期化完了 (LCD電源+ES8311電源ON)");
}

/* --------------------------------------------------------
 * ES8311 コーデック初期化 (esp_codec_dev + 新I2C API)
 * IDF 5.3では pmic_lcd_power_on() で初期化済みの
 * s_i2c_bus_handle を再利用する
 * -------------------------------------------------------- */
static void codec_init(void) {
    /* I2Cバスは pmic_lcd_power_on() で初期化済みのはず */
    i2c_master_bus_handle_t i2c_bus_handle = s_i2c_bus_handle;
    if (!i2c_bus_handle) {
        /* 未初期化の場合は新規初期化 */
        i2c_master_bus_config_t i2c_bus_cfg = {
            .clk_source        = I2C_CLK_SRC_DEFAULT,
            .i2c_port          = I2C_NUM_0,
            .sda_io_num        = I2C_SDA_GPIO,
            .scl_io_num        = I2C_SCL_GPIO,
            .glitch_ignore_cnt = 7,
            .flags.enable_internal_pullup = true,
        };
        ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_bus_cfg, &i2c_bus_handle));
        s_i2c_bus_handle = i2c_bus_handle;
    }

    /* まずES8311が存在するか確認する */
    ESP_LOGI(TAG, "I2Cスキャン (SDA=%d, SCL=%d)...", I2C_SDA_GPIO, I2C_SCL_GPIO);
    bool es8311_found = false;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        esp_err_t probe = i2c_master_probe(i2c_bus_handle, addr, pdMS_TO_TICKS(10));
        if (probe == ESP_OK) {
            const char *label = "";
            if (addr == ES8311_I2C_ADDR)  label = " ← ES8311";
            else if (addr == PMIC_I2C_ADDR) label = " ← PMIC(M5PM1)";
            else if (addr == 0x68)          label = " ← BMI270(IMU)";
            ESP_LOGI(TAG, "  I2Cデバイス: 0x%02X%s", addr, label);
            if (addr == ES8311_I2C_ADDR) es8311_found = true;
        }
    }

    if (!es8311_found) {
        ESP_LOGW(TAG, "ES8311 (0x%02X) 未検出 - コーデックなしで継続", ES8311_I2C_ADDR);
        return;
    }

    /* esp_codec_devのI2Cインターフェースは内部でデバイスハンドルを作成するが
     * そのハンドル経由の書き込みがNACKになる (ESP-IDF 5.x新API互換性問題)。
     * 直接 i2c_master_transmit は動作するため、M5Unifiedと同じ方法で
     * ES8311レジスタを直接初期化する。 */

    /* ES8311用I2Cデバイスハンドル作成 */
    static i2c_master_dev_handle_t s_es8311_dev = NULL;
    i2c_device_config_t es_dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = ES8311_I2C_ADDR,
        .scl_speed_hz    = 100000,
    };
    if (i2c_master_bus_add_device(i2c_bus_handle, &es_dev_cfg, &s_es8311_dev) != ESP_OK) {
        ESP_LOGE(TAG, "ES8311デバイスハンドル作成失敗");
        return;
    }

    /* ES8311レジスタ直接初期化 (M5Unified解析結果に基づく)
     * スピーカー(DAC) + マイク(ADC) 両方を有効化 */
    typedef struct { uint8_t reg; uint8_t val; } es8311_reg_t;
    static const es8311_reg_t init_seq[] = {
        /* リセット & クロック設定 */
        {0x00, 0x1F},  /* Reset all registers */
        {0x00, 0x80},  /* CSM power on, reset off */
        {0x01, 0x3F},  /* MCLK from BCLK, MCLK active, BCLK active */
        {0x02, 0x00},  /* CLK_MGR: divider = 1 */
        {0x03, 0x10},  /* ADC_OSR = 32x */
        {0x04, 0x10},  /* DAC_OSR = 32x */
        {0x05, 0x00},  /* CLK on */
        /* システム設定 */
        {0x0B, 0x00},  /* System: normal */
        {0x0D, 0x01},  /* Power up analog circuitry */
        {0x0E, 0x02},  /* Enable analog PGA, enable ADC modulator */
        {0x0F, 0x44},  /* ADC: I2S 16bit */
        {0x10, 0x0C},  /* DAC: I2S 16bit */
        {0x11, 0x00},  /* ADC/DAC: normal operation */
        /* ADC設定 */
        {0x12, 0x00},  /* Power up DAC */
        {0x13, 0x10},  /* Enable output to HP drive */
        {0x14, 0x18},  /* ADC: MIC1P-MIC1N, PGA gain +24dB */
        {0x17, 0xBF},  /* ADC volume */
        {0x1C, 0x6A},  /* ADC equalizer bypass, DC offset cancel */
        /* DAC設定 */
        {0x32, 0xBF},  /* DAC volume (0dB) */
        {0x37, 0x08},  /* DAC bypass equalizer */
    };
    bool init_ok = true;
    for (size_t i = 0; i < sizeof(init_seq) / sizeof(init_seq[0]); i++) {
        uint8_t buf[2] = { init_seq[i].reg, init_seq[i].val };
        esp_err_t ret = i2c_master_transmit(s_es8311_dev, buf, 2, pdMS_TO_TICKS(100));
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "ES8311 reg0x%02X=0x%02X 失敗: %s",
                     init_seq[i].reg, init_seq[i].val, esp_err_to_name(ret));
            init_ok = false;
        }
        /* リセット後は少し待つ */
        if (init_seq[i].reg == 0x00 && init_seq[i].val == 0x1F) {
            vTaskDelay(pdMS_TO_TICKS(20));
        }
    }
    if (init_ok) {
        ESP_LOGI(TAG, "ES8311 直接初期化完了 (全レジスタOK)");
    } else {
        ESP_LOGW(TAG, "ES8311 一部レジスタ書き込み失敗あり");
    }

    s_codec_available = true;
    ESP_LOGI(TAG, "ES8311 コーデック初期化完了");
}

/* --------------------------------------------------------
 * 簡易マイク入力タスク (Core 1)
 * I2S RX → エネルギーベースVAD → コールバック
 * ESP-SR AFEなしの軽量版
 * -------------------------------------------------------- */
static void mic_raw_task(void *arg) {
    int16_t *mic_buf = malloc(AUDIO_CHUNK_BYTES);
    if (!mic_buf) {
        ESP_LOGE(TAG, "マイクバッファ確保失敗");
        vTaskDelete(NULL);
        return;
    }

    bool prev_speech = false;
    int speech_count = 0;
    int silence_count = 0;
    int log_counter = 0;

    while (true) {
        size_t bytes_read = 0;
        i2s_channel_read(s_rx_handle, mic_buf, AUDIO_CHUNK_BYTES,
                         &bytes_read, portMAX_DELAY);

        size_t sample_count = bytes_read / sizeof(int16_t);

        /* エネルギー計算 (RMS²: 平方根は取らず二乗のまま比較) */
        int64_t energy = 0;
        for (size_t i = 0; i < sample_count; i++) {
            energy += (int64_t)mic_buf[i] * mic_buf[i];
        }
        int64_t rms_sq = 0;
        if (sample_count > 0) {
            rms_sq = energy / sample_count;
        }

        bool is_speech = (rms_sq > VAD_RMS_SQ_THRESHOLD);

        /* デバッグ: 5秒ごとにRMS²値をログ出力 */
        if (++log_counter >= 155) {
            log_counter = 0;
            ESP_LOGI(TAG, "VAD: rms²=%lld speech=%d",
                     (long long)rms_sq, (int)prev_speech);
        }

        if (is_speech) {
            speech_count++;
            silence_count = 0;
            if (speech_count >= VAD_SPEECH_FRAMES && !prev_speech) {
                prev_speech = true;
                ESP_LOGI(TAG, "VAD: 発話開始検出");
                if (s_vad_cb) s_vad_cb(true);
            }
        } else {
            silence_count++;
            speech_count = 0;
            if (silence_count >= VAD_SILENCE_FRAMES && prev_speech) {
                prev_speech = false;
                ESP_LOGI(TAG, "VAD: 無音検出");
                if (s_vad_cb) s_vad_cb(false);
            }
        }

        /* クリーン音声コールバック (生マイクデータ) */
        if (s_rx_cb && sample_count > 0) {
            s_rx_cb(mic_buf, sample_count);
        }
    }
}

/* --------------------------------------------------------
 * TTS再生タスク (Core 0)
 * 再生バッファ → I2S TX
 * -------------------------------------------------------- */
static void playback_task(void *arg) {
    static int16_t silent_buf[AUDIO_CHUNK_SAMPLES];
    memset(silent_buf, 0, sizeof(silent_buf));

    size_t bytes_written = 0;

    while (true) {
        if (s_stop_playback) {
            /* 停止: バッファをフラッシュ */
            void *item;
            size_t item_size;
            while ((item = xRingbufferReceive(
                        s_playback_buf, &item_size, pdMS_TO_TICKS(0))) != NULL) {
                vRingbufferReturnItem(s_playback_buf, item);
            }
            s_stop_playback = false;
        }

        size_t received = 0;
        void *data = xRingbufferReceiveUpTo(
            s_playback_buf, &received, pdMS_TO_TICKS(20), AUDIO_CHUNK_BYTES);

        if (data && received > 0) {
            i2s_channel_write(s_tx_handle, data, received,
                              &bytes_written, pdMS_TO_TICKS(100));
            vRingbufferReturnItem(s_playback_buf, data);
        } else {
            /* 無音を送信して I2S クロックを維持 */
            i2s_channel_write(s_tx_handle, silent_buf, AUDIO_CHUNK_BYTES,
                              &bytes_written, pdMS_TO_TICKS(20));
        }
    }
}

/* --------------------------------------------------------
 * 公開 API
 * -------------------------------------------------------- */

void audio_hal_init(audio_rx_callback_t rx_cb, vad_event_callback_t vad_cb) {
    s_rx_cb  = rx_cb;
    s_vad_cb = vad_cb;

    s_aec_ref_buf  = xRingbufferCreate(AEC_REF_RINGBUF_SIZE,
                                        RINGBUF_TYPE_BYTEBUF);
    s_playback_buf = xRingbufferCreate(PLAYBACK_RINGBUF_SIZE,
                                        RINGBUF_TYPE_BYTEBUF);
    if (!s_aec_ref_buf || !s_playback_buf) {
        ESP_LOGE(TAG, "リングバッファ確保失敗");
        return;
    }

    i2s_init();
    codec_init();

    /* 再生タスク (Core 0) */
    xTaskCreatePinnedToCore(playback_task, "playback",
                             4096, NULL, 5, NULL, 0);

    /* マイク入力タスク (Core 1): I2S RX → コールバック (AFEなし簡易版) */
    xTaskCreatePinnedToCore(mic_raw_task, "mic_raw",
                             4096, NULL, 5, NULL, 1);

    ESP_LOGI(TAG, "音声HAL初期化完了 (簡易VAD, AFEなし)");
}

void audio_hal_play_bytes(const uint8_t *data, size_t len) {
    if (!data || len == 0) return;

    if (xRingbufferSend(s_playback_buf, data, len,
                        pdMS_TO_TICKS(100)) != pdTRUE) {
        ESP_LOGW(TAG, "再生バッファ満杯: %zu バイト破棄", len);
        return;
    }

    /* AEC リファレンスにも同じデータを供給 */
    xRingbufferSend(s_aec_ref_buf, data, len, pdMS_TO_TICKS(10));
}

void audio_hal_stop_playback(void) {
    s_stop_playback = true;
    ESP_LOGI(TAG, "再生停止要求");
}
