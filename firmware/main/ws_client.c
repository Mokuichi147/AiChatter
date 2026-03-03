#include "ws_client.h"

#include <stdlib.h>
#include <string.h>

#include "audio_hal.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lcd_display.h"
#include "state_machine.h"

#define TAG "WS_CLIENT"
#define HEADER_SIZE 7  // [type:1][seq_hi:1][seq_lo:1][len:4]
#define RECONNECT_CHECK_INTERVAL_MS  5000  /* 接続監視間隔 */
#define RECONNECT_FORCE_AFTER_MS    15000  /* この期間未接続なら強制再接続 */

/* Server -> ESP32 メッセージ */
#define MSG_TTS_CHUNK            0x02
#define MSG_TTS_END              0x03
#define MSG_SLEEP                0x04
#define MSG_WAKE                 0x05
#define MSG_DISPLAY_TEXT         0x20
#define MSG_DISPLAY_IMAGE_BLOCK  0x21

#define DISPLAY_TEXT_META_SIZE   6
#define DISPLAY_IMAGE_META_SIZE  8

static esp_websocket_client_handle_t s_client = NULL;
static tts_audio_callback_t s_tts_cb = NULL;
static tts_end_callback_t s_end_cb = NULL;
static uint16_t s_seq = 0;
static uint8_t s_current_msg_type = 0;  /* フレーム分割配信時のメッセージタイプ保持 */
static volatile bool s_started = false;  /* クライアント起動済みフラグ */

/* オフライン音声バッファ (PSRAM) — 約8秒分 */
#define AUDIO_BUF_SIZE       (256 * 1024)
static uint8_t *s_audio_buf = NULL;
static size_t   s_audio_buf_len = 0;
static SemaphoreHandle_t s_audio_buf_mutex = NULL;

/* フレーム再組立バッファ (分割配信対策) */
#define REASSEMBLY_BUF_SIZE  8192
static uint8_t s_reassembly_buf[REASSEMBLY_BUF_SIZE];
static size_t  s_reassembly_len = 0;
static size_t  s_reassembly_expected = 0;  /* ヘッダから読んだペイロード長 */

/* ヘッダー生成: ビッグエンディアン */
static void make_header(uint8_t *buf, uint8_t type, uint16_t seq,
                        uint32_t payload_len) {
    buf[0] = type;
    buf[1] = (seq >> 8) & 0xFF;
    buf[2] = seq & 0xFF;
    buf[3] = (payload_len >> 24) & 0xFF;
    buf[4] = (payload_len >> 16) & 0xFF;
    buf[5] = (payload_len >> 8) & 0xFF;
    buf[6] = payload_len & 0xFF;
}

static uint16_t read_u16_be(const uint8_t *p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

static void reset_reassembly(void) {
    s_current_msg_type = 0;
    s_reassembly_len = 0;
    s_reassembly_expected = 0;
}

static void handle_display_text(const uint8_t *payload, size_t len) {
    if (len < DISPLAY_TEXT_META_SIZE) {
        ESP_LOGW(TAG, "MSG_DISPLAY_TEXTのペイロードが短すぎます: %u",
                 (unsigned)len);
        return;
    }

    uint8_t size = payload[0];
    bool clear = payload[1] != 0;
    uint16_t x = read_u16_be(payload + 2);
    uint16_t y = read_u16_be(payload + 4);

    size_t text_len = len - DISPLAY_TEXT_META_SIZE;
    if (text_len > 512) text_len = 512;

    char text[513];
    memcpy(text, payload + DISPLAY_TEXT_META_SIZE, text_len);
    text[text_len] = '\0';

    lcd_show_text(text, size, x, y, clear);
}

static void handle_display_image_block(const uint8_t *payload, size_t len) {
    if (len < DISPLAY_IMAGE_META_SIZE) {
        ESP_LOGW(TAG, "MSG_DISPLAY_IMAGE_BLOCKのペイロードが短すぎます: %u",
                 (unsigned)len);
        return;
    }

    uint16_t x = read_u16_be(payload + 0);
    uint16_t y = read_u16_be(payload + 2);
    uint16_t w = read_u16_be(payload + 4);
    uint16_t h = read_u16_be(payload + 6);

    const uint8_t *img = payload + DISPLAY_IMAGE_META_SIZE;
    size_t img_len = len - DISPLAY_IMAGE_META_SIZE;
    lcd_draw_rgb565(x, y, w, h, img, img_len);
}

static void handle_payload_message(uint8_t msg_type, const uint8_t *payload,
                                   size_t payload_len) {
    switch (msg_type) {
        case MSG_TTS_CHUNK:
            if (s_tts_cb && payload_len > 0) {
                s_tts_cb(payload, payload_len);
            }
            break;
        case MSG_DISPLAY_TEXT:
            handle_display_text(payload, payload_len);
            break;
        case MSG_DISPLAY_IMAGE_BLOCK:
            handle_display_image_block(payload, payload_len);
            break;
        default:
            break;
    }
}

/* バッファ音声をWSで送信 (再接続時に呼び出し) */
static void flush_audio_buffer(void) {
    if (!s_audio_buf || !s_audio_buf_mutex) return;
    if (xSemaphoreTake(s_audio_buf_mutex, pdMS_TO_TICKS(100)) != pdTRUE) return;

    if (s_audio_buf_len > 0 && esp_websocket_client_is_connected(s_client)) {
        ESP_LOGI(TAG, "バッファ音声フラッシュ: %u bytes", (unsigned)s_audio_buf_len);
        size_t offset = 0;
        while (offset < s_audio_buf_len) {
            size_t remaining = s_audio_buf_len - offset;
            size_t chunk = remaining > 4096 ? 4096 : remaining;
            size_t total = HEADER_SIZE + chunk;
            uint8_t *buf = malloc(total);
            if (!buf) break;
            make_header(buf, 0x01, ++s_seq, chunk);
            memcpy(buf + HEADER_SIZE, s_audio_buf + offset, chunk);
            int ret = esp_websocket_client_send_bin(
                s_client, (const char *)buf, total, pdMS_TO_TICKS(1000));
            free(buf);
            if (ret < 0) break;
            offset += chunk;
        }
    }
    s_audio_buf_len = 0;
    xSemaphoreGive(s_audio_buf_mutex);
}

static void ws_event_handler(void *handler_args, esp_event_base_t base,
                              int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data =
        (esp_websocket_event_data_t *)event_data;

    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "WebSocket接続完了");
            s_reassembly_len = 0;
            s_reassembly_expected = 0;
            flush_audio_buffer();
            state_machine_post_event(SM_EVENT_WS_CONNECTED, NULL, 0);
            break;

        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "WebSocket切断 (自動再接続待機中...)");
            state_machine_post_event(SM_EVENT_WS_DISCONNECTED, NULL, 0);
            break;

        case WEBSOCKET_EVENT_ERROR:
            ESP_LOGE(TAG, "WebSocketエラー (自動再接続待機中...)");
            state_machine_post_event(SM_EVENT_WS_DISCONNECTED, NULL, 0);
            break;

        case WEBSOCKET_EVENT_DATA:
            /* バイナリフレームのみ処理 (op_code=2) */
            if (data->op_code != 2) break;

            {
                const uint8_t *buf = (const uint8_t *)data->data_ptr;
                size_t len = data->data_len;

                /* フレーム分割配信対策:
                 * esp_websocket_clientはbuffer_sizeより大きなフレームや
                 * TCP分割により複数イベントに分けて配信することがある。
                 * 再組立バッファで完全なメッセージを構築してから処理する。 */

                if (data->payload_offset == 0) {
                    /* 新しいフレーム開始 */
                    reset_reassembly();

                    if (len < HEADER_SIZE) break;

                    uint8_t msg_type = buf[0];
                    s_current_msg_type = msg_type;

                    uint32_t payload_len = ((uint32_t)buf[3] << 24) |
                                           ((uint32_t)buf[4] << 16) |
                                           ((uint32_t)buf[5] << 8) | buf[6];

                    if (msg_type == MSG_TTS_END) {
                        /* TTS終了 (ペイロードなし) */
                        audio_hal_notify_tts_end();
                        if (s_end_cb) s_end_cb();
                        reset_reassembly();
                        break;
                    }

                    if (msg_type == MSG_SLEEP) {
                        /* スリープ指示 */
                        ESP_LOGI(TAG, "MSG_SLEEP受信");
                        state_machine_post_event(SM_EVENT_SLEEP, NULL, 0);
                        reset_reassembly();
                        break;
                    }

                    if (msg_type == MSG_WAKE) {
                        /* ウェイク指示 */
                        ESP_LOGI(TAG, "MSG_WAKE受信");
                        state_machine_post_event(SM_EVENT_WAKE, NULL, 0);
                        reset_reassembly();
                        break;
                    }

                    if (msg_type != MSG_TTS_CHUNK &&
                        msg_type != MSG_DISPLAY_TEXT &&
                        msg_type != MSG_DISPLAY_IMAGE_BLOCK) {
                        break;
                    }

                    if (payload_len > REASSEMBLY_BUF_SIZE) {
                        ESP_LOGW(TAG, "受信ペイロードが大きすぎます: type=0x%02X len=%u",
                                 msg_type, (unsigned)payload_len);
                        reset_reassembly();
                        break;
                    }

                    /* ペイロードを再組立バッファにコピー */
                    size_t avail = len - HEADER_SIZE;
                    size_t copy = avail;
                    if (copy > REASSEMBLY_BUF_SIZE) copy = REASSEMBLY_BUF_SIZE;
                    memcpy(s_reassembly_buf, buf + HEADER_SIZE, copy);
                    s_reassembly_len = copy;
                    s_reassembly_expected = payload_len;

                    /* フレームが1イベントで完結した場合は即送信 */
                    if (s_reassembly_len >= s_reassembly_expected) {
                        handle_payload_message(s_current_msg_type, s_reassembly_buf,
                                               s_reassembly_len);
                        reset_reassembly();
                    }
                } else {
                    /* 分割フレームの続き */
                    if (s_reassembly_expected == 0) break;

                    size_t remain = REASSEMBLY_BUF_SIZE - s_reassembly_len;
                    size_t copy = len;
                    if (copy > remain) copy = remain;
                    memcpy(s_reassembly_buf + s_reassembly_len, buf, copy);
                    s_reassembly_len += copy;

                    /* 全データ受信完了 */
                    if (s_reassembly_len >= s_reassembly_expected) {
                        handle_payload_message(s_current_msg_type, s_reassembly_buf,
                                               s_reassembly_len);
                        reset_reassembly();
                    }
                }
            }
            break;

        default:
            break;
    }
}

/* 接続監視タスク: 未接続が一定期間続いたらstop→startで強制再接続 */
static void ws_reconnect_task(void *arg) {
    TickType_t disconnect_since = 0;

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(RECONNECT_CHECK_INTERVAL_MS));
        if (!s_client || !s_started) continue;

        if (esp_websocket_client_is_connected(s_client)) {
            disconnect_since = 0;
            continue;
        }

        /* 未接続状態の開始を記録 */
        if (disconnect_since == 0) {
            disconnect_since = xTaskGetTickCount();
            continue;
        }

        TickType_t elapsed = xTaskGetTickCount() - disconnect_since;
        if (elapsed >= pdMS_TO_TICKS(RECONNECT_FORCE_AFTER_MS)) {
            ESP_LOGW(TAG, "長時間未接続 → 強制再接続");
            esp_websocket_client_stop(s_client);
            vTaskDelay(pdMS_TO_TICKS(500));
            esp_websocket_client_start(s_client);
            disconnect_since = 0;
        }
    }
}

void ws_client_init(const char *uri, tts_audio_callback_t tts_cb,
                    tts_end_callback_t end_cb) {
    s_tts_cb = tts_cb;
    s_end_cb = end_cb;

    /* オフライン音声バッファをPSRAMに確保 */
    s_audio_buf = heap_caps_malloc(AUDIO_BUF_SIZE, MALLOC_CAP_SPIRAM);
    if (s_audio_buf) {
        ESP_LOGI(TAG, "音声バッファ確保: %d KB (PSRAM)", AUDIO_BUF_SIZE / 1024);
    } else {
        ESP_LOGW(TAG, "PSRAM音声バッファ確保失敗");
    }
    s_audio_buf_mutex = xSemaphoreCreateMutex();

    esp_websocket_client_config_t cfg = {
        .uri = uri,
        .reconnect_timeout_ms = 1000,
        .network_timeout_ms = 10000,
        .buffer_size = 65536,
        .ping_interval_sec = 10,
    };

    s_client = esp_websocket_client_init(&cfg);
    esp_websocket_register_events(s_client, WEBSOCKET_EVENT_ANY,
                                  ws_event_handler, NULL);
    esp_websocket_client_start(s_client);
    s_started = true;

    xTaskCreate(ws_reconnect_task, "ws_reconn", 2048, NULL, 3, NULL);

    ESP_LOGI(TAG, "WebSocketクライアント起動: %s", uri);
}

void ws_client_send_audio(const int16_t *samples, size_t count) {
    size_t payload_len = count * sizeof(int16_t);

    if (!s_client || !esp_websocket_client_is_connected(s_client)) {
        /* 未接続: PSRAMバッファに蓄積 */
        if (s_audio_buf && s_audio_buf_mutex &&
            xSemaphoreTake(s_audio_buf_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            if (s_audio_buf_len + payload_len <= AUDIO_BUF_SIZE) {
                memcpy(s_audio_buf + s_audio_buf_len, samples, payload_len);
                s_audio_buf_len += payload_len;
            }
            xSemaphoreGive(s_audio_buf_mutex);
        }
        return;
    }

    size_t total_len = HEADER_SIZE + payload_len;
    uint8_t *buf = malloc(total_len);
    if (!buf) {
        ESP_LOGE(TAG, "メモリ確保失敗");
        return;
    }

    make_header(buf, 0x01, ++s_seq, payload_len);
    memcpy(buf + HEADER_SIZE, samples, payload_len);

    int ret = esp_websocket_client_send_bin(s_client, (const char *)buf,
                                             total_len, pdMS_TO_TICKS(1000));
    if (ret < 0) {
        ESP_LOGW(TAG, "音声送信失敗");
    }

    free(buf);
}

void ws_client_send_end_of_speech(void) {
    if (!s_client || !esp_websocket_client_is_connected(s_client)) {
        return;
    }

    uint8_t buf[HEADER_SIZE];
    make_header(buf, 0x11, ++s_seq, 0);
    esp_websocket_client_send_bin(s_client, (const char *)buf, HEADER_SIZE,
                                  pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "EOS送信 (seq=%u)", s_seq);
}

void ws_client_send_button(void) {
    if (!s_client || !esp_websocket_client_is_connected(s_client)) {
        return;
    }

    uint8_t buf[HEADER_SIZE];
    make_header(buf, 0x13, ++s_seq, 0);
    esp_websocket_client_send_bin(s_client, (const char *)buf, HEADER_SIZE,
                                  pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "ボタン押下通知送信 (seq=%u)", s_seq);
}

void ws_client_send_interrupt(void) {
    if (!s_client || !esp_websocket_client_is_connected(s_client)) {
        return;
    }

    uint8_t buf[HEADER_SIZE];
    make_header(buf, 0x12, ++s_seq, 0);
    esp_websocket_client_send_bin(s_client, (const char *)buf, HEADER_SIZE,
                                  pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "バージイン割り込み送信 (seq=%u)", s_seq);
}

bool ws_client_is_connected(void) {
    return s_client != NULL && esp_websocket_client_is_connected(s_client);
}

void ws_client_clear_audio_buffer(void) {
    if (s_audio_buf_mutex &&
        xSemaphoreTake(s_audio_buf_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        s_audio_buf_len = 0;
        xSemaphoreGive(s_audio_buf_mutex);
    }
}
