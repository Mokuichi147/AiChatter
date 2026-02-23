#include "ws_client.h"

#include <stdlib.h>
#include <string.h>

#include "esp_log.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "WS_CLIENT"
#define HEADER_SIZE 7  // [type:1][seq_hi:1][seq_lo:1][len:4]

static esp_websocket_client_handle_t s_client = NULL;
static tts_audio_callback_t s_tts_cb = NULL;
static tts_end_callback_t s_end_cb = NULL;
static uint16_t s_seq = 0;

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

static void ws_event_handler(void *handler_args, esp_event_base_t base,
                              int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data =
        (esp_websocket_event_data_t *)event_data;

    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "WebSocket接続完了");
            break;

        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "WebSocket切断 (自動再接続待機中...)");
            break;

        case WEBSOCKET_EVENT_ERROR:
            ESP_LOGE(TAG, "WebSocketエラー");
            break;

        case WEBSOCKET_EVENT_DATA:
            /* バイナリフレームのみ処理 (op_code=2) */
            if (data->op_code != 2 || data->data_len < HEADER_SIZE) {
                break;
            }

            {
                const uint8_t *buf = (const uint8_t *)data->data_ptr;
                uint8_t msg_type = buf[0];
                uint32_t payload_len = ((uint32_t)buf[3] << 24) |
                                       ((uint32_t)buf[4] << 16) |
                                       ((uint32_t)buf[5] << 8) | buf[6];

                if (msg_type == 0x02) {
                    /* TTS音声チャンク */
                    if (s_tts_cb && payload_len > 0) {
                        s_tts_cb(buf + HEADER_SIZE, payload_len);
                    }
                } else if (msg_type == 0x03) {
                    /* TTS終了 */
                    if (s_end_cb) {
                        s_end_cb();
                    }
                } else {
                    ESP_LOGD(TAG, "不明なサーバーメッセージ: 0x%02X", msg_type);
                }
            }
            break;

        default:
            break;
    }
}

void ws_client_init(const char *uri, tts_audio_callback_t tts_cb,
                    tts_end_callback_t end_cb) {
    s_tts_cb = tts_cb;
    s_end_cb = end_cb;

    esp_websocket_client_config_t cfg = {
        .uri = uri,
        .reconnect_timeout_ms = 3000,
        .network_timeout_ms = 10000,
        .buffer_size = 65536,
    };

    s_client = esp_websocket_client_init(&cfg);
    esp_websocket_register_events(s_client, WEBSOCKET_EVENT_ANY,
                                  ws_event_handler, NULL);
    esp_websocket_client_start(s_client);

    ESP_LOGI(TAG, "WebSocketクライアント起動: %s", uri);
}

void ws_client_send_audio(const int16_t *samples, size_t count) {
    if (!s_client || !esp_websocket_client_is_connected(s_client)) {
        return;
    }

    size_t payload_len = count * sizeof(int16_t);
    size_t total_len = HEADER_SIZE + payload_len;

    uint8_t *buf = malloc(total_len);
    if (!buf) {
        ESP_LOGE(TAG, "メモリ確保失敗");
        return;
    }

    make_header(buf, 0x01, ++s_seq, payload_len);
    memcpy(buf + HEADER_SIZE, samples, payload_len);

    int ret = esp_websocket_client_send_bin(s_client, (const char *)buf,
                                             total_len, pdMS_TO_TICKS(200));
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
