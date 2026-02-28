#include "ws_client.h"

#include <stdlib.h>
#include <string.h>

#include "audio_hal.h"
#include "esp_log.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "state_machine.h"

#define TAG "WS_CLIENT"
#define HEADER_SIZE 7  // [type:1][seq_hi:1][seq_lo:1][len:4]

static esp_websocket_client_handle_t s_client = NULL;
static tts_audio_callback_t s_tts_cb = NULL;
static tts_end_callback_t s_end_cb = NULL;
static uint16_t s_seq = 0;
static uint8_t s_current_msg_type = 0;  /* フレーム分割配信時のメッセージタイプ保持 */

/* フレーム再組立バッファ (分割配信対策) */
#define REASSEMBLY_BUF_SIZE  (4096 + HEADER_SIZE)
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
                    s_reassembly_len = 0;
                    s_reassembly_expected = 0;

                    if (len < HEADER_SIZE) break;

                    uint8_t msg_type = buf[0];
                    s_current_msg_type = msg_type;

                    uint32_t payload_len = ((uint32_t)buf[3] << 24) |
                                           ((uint32_t)buf[4] << 16) |
                                           ((uint32_t)buf[5] << 8) | buf[6];

                    if (msg_type == 0x03) {
                        /* TTS終了 (ペイロードなし) */
                        audio_hal_notify_tts_end();
                        if (s_end_cb) s_end_cb();
                        break;
                    }

                    if (msg_type == 0x04) {
                        /* スリープ指示 */
                        ESP_LOGI(TAG, "MSG_SLEEP受信");
                        state_machine_post_event(SM_EVENT_SLEEP, NULL, 0);
                        break;
                    }

                    if (msg_type == 0x05) {
                        /* ウェイク指示 */
                        ESP_LOGI(TAG, "MSG_WAKE受信");
                        state_machine_post_event(SM_EVENT_WAKE, NULL, 0);
                        break;
                    }

                    if (msg_type != 0x02) break;

                    /* ペイロードを再組立バッファにコピー */
                    size_t avail = len - HEADER_SIZE;
                    size_t copy = avail;
                    if (copy > REASSEMBLY_BUF_SIZE) copy = REASSEMBLY_BUF_SIZE;
                    memcpy(s_reassembly_buf, buf + HEADER_SIZE, copy);
                    s_reassembly_len = copy;
                    s_reassembly_expected = payload_len;

                    /* フレームが1イベントで完結した場合は即送信 */
                    if (s_reassembly_len >= s_reassembly_expected) {
                        if (s_tts_cb && s_reassembly_len > 0) {
                            s_tts_cb(s_reassembly_buf, s_reassembly_len);
                        }
                        s_reassembly_len = 0;
                        s_reassembly_expected = 0;
                    }
                } else {
                    /* 分割フレームの続き */
                    if (s_current_msg_type != 0x02 || s_reassembly_expected == 0) break;

                    size_t remain = REASSEMBLY_BUF_SIZE - s_reassembly_len;
                    size_t copy = len;
                    if (copy > remain) copy = remain;
                    memcpy(s_reassembly_buf + s_reassembly_len, buf, copy);
                    s_reassembly_len += copy;

                    /* 全データ受信完了 */
                    if (s_reassembly_len >= s_reassembly_expected) {
                        if (s_tts_cb && s_reassembly_len > 0) {
                            s_tts_cb(s_reassembly_buf, s_reassembly_len);
                        }
                        s_reassembly_len = 0;
                        s_reassembly_expected = 0;
                    }
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
        .network_timeout_ms = 120000,
        .buffer_size = 65536,
        .ping_interval_sec = 15,
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
