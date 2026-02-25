#include <stdio.h>
#include <string.h>

#include "audio_hal.h"
#include "config.h"
#include "driver/gpio.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lcd_display.h"
#include "nvs_flash.h"
#include "state_machine.h"
#include "ws_client.h"

#define TAG "MAIN"

/* WiFi接続完了イベントビット */
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

#define WIFI_MAX_RETRIES    10
static int s_retry_count = 0;

/* ============================================================
 * WiFi イベントハンドラ
 * ============================================================ */
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT &&
               event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_count < WIFI_MAX_RETRIES) {
            esp_wifi_connect();
            s_retry_count++;
            ESP_LOGI(TAG, "WiFi再接続中... (%d/%d)", s_retry_count,
                     WIFI_MAX_RETRIES);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
            ESP_LOGE(TAG, "WiFi接続失敗");
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "IPアドレス取得: " IPSTR, IP2STR(&event->ip_info.ip));
        lcd_log("IP:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* WiFiステーションモードで接続する */
static bool wifi_connect(void) {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL,
        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL,
        &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "WiFi接続中... SSID: %s", WIFI_SSID);

    /* 接続完了またはタイムアウト待機 */
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE,
        pdMS_TO_TICKS(15000));

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "WiFi接続完了");
        return true;
    } else {
        ESP_LOGE(TAG, "WiFi接続失敗 (SSID/パスワードを確認してください)");
        return false;
    }
}

/* ============================================================
 * 音声コールバック
 * ============================================================ */

/**
 * AFEからのクリーン音声受信コールバック
 * LISTENING / VAD_SILENCE 状態のときのみサーバーへ送信
 */
static void audio_rx_callback(const int16_t *samples, size_t count) {
    sm_state_t state = state_machine_get_state();
    if (state == SM_STATE_LISTENING || state == SM_STATE_VAD_SILENCE) {
        ws_client_send_audio(samples, count);
    }
}

/**
 * VAD検出コールバック
 */
static void vad_event_callback(bool voice_detected) {
    if (voice_detected) {
        state_machine_post_event(SM_EVENT_VAD_START, NULL, 0);
    } else {
        state_machine_post_event(SM_EVENT_VAD_END, NULL, 0);
    }
}

/* ============================================================
 * WebSocketコールバック
 * ============================================================ */

/**
 * TTS音声チャンク受信コールバック (WebSocket)
 * 音声データは直接再生バッファへ、ステートマシンには通知のみ
 */
static void tts_audio_callback(const uint8_t *data, size_t len) {
    audio_hal_play_bytes(data, len);
    state_machine_post_event(SM_EVENT_WS_TTS_CHUNK, NULL, 0);
}

/**
 * TTS終了コールバック (WebSocket)
 */
static void tts_end_callback(void) {
    state_machine_post_event(SM_EVENT_WS_TTS_END, NULL, 0);
}

/* ============================================================
 * ボタン処理タスク
 * ボタンAの長押し (1秒) でリセット/再接続
 * ============================================================ */
static void button_task(void *arg) {
    /* ボタンAは内蔵プルアップ付き、押下でLOW */
    gpio_config_t btn_cfg = {
        .pin_bit_mask = (1ULL << BUTTON_A_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&btn_cfg);

    bool was_pressed = false;
    uint32_t press_start = 0;

    while (true) {
        bool pressed = (gpio_get_level(BUTTON_A_GPIO) == 0);

        if (pressed && !was_pressed) {
            press_start = xTaskGetTickCount();
            was_pressed = true;
            ESP_LOGD(TAG, "ボタンA 押下");
        } else if (!pressed && was_pressed) {
            was_pressed = false;
            uint32_t duration_ms =
                (xTaskGetTickCount() - press_start) * portTICK_PERIOD_MS;
            ESP_LOGD(TAG, "ボタンA 離した (%ldms)", (long)duration_ms);

            /* 1秒長押し: バージイン強制 (デバッグ用) */
            if (duration_ms > 1000) {
                ESP_LOGI(TAG, "長押し: 強制割り込み");
                sm_state_t state = state_machine_get_state();
                if (state == SM_STATE_SPEAKING) {
                    audio_hal_stop_playback();
                    ws_client_send_interrupt();
                    state_machine_post_event(SM_EVENT_VAD_START, NULL, 0);
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/* ============================================================
 * app_main
 * ============================================================ */
void app_main(void) {
    ESP_LOGI(TAG, "AiChatter 起動中...");

    /* NVS初期化 */

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* USB列挙待ち (デバッグ用: シリアルモニター接続時間確保) */
    vTaskDelay(pdMS_TO_TICKS(2000));

    /* PMIC初期化 (LCD電源・ES8311電源有効化) - lcd_init()より前に必須 */
    pmic_lcd_power_on();

    /* LCD初期化 (最初に起動して視覚的フィードバックを提供) */
    lcd_init();

    /* ステートマシン初期化 */
    state_machine_init();

    /* WiFi接続 */
    lcd_log("WiFi: %s", WIFI_SSID);
    if (!wifi_connect()) {
        ESP_LOGE(TAG, "WiFi接続失敗");
        lcd_log("WiFi FAIL!");
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
    lcd_log("WiFi OK");

    /* 音声HAL初期化 */
    lcd_log("Audio init...");
    audio_hal_init(audio_rx_callback, vad_event_callback);
    lcd_log("Audio OK");

    /* WebSocketクライアント初期化 */
    lcd_log("WS->%s", WS_SERVER_URI + 5);  /* "ws://" を省略 */
    ws_client_init(WS_SERVER_URI, tts_audio_callback, tts_end_callback);

    /* ボタンタスク起動 */
    xTaskCreatePinnedToCore(button_task, "button", 4096, NULL, 3, NULL, 0);

    ESP_LOGI(TAG, "AiChatter 起動完了 - 話しかけてください");
    ESP_LOGI(TAG, "サーバー: %s", WS_SERVER_URI);
    lcd_log("Ready!");

    /* メインループ: ウォッチドッグ用にスリープ */
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGD(TAG, "ヒープ: %lu bytes free",
                 (unsigned long)esp_get_free_heap_size());
    }
}
