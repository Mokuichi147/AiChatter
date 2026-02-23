#include "state_machine.h"

#include <string.h>

#include "audio_hal.h"
#include "config.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "lcd_display.h"
#include "ws_client.h"

#define TAG "STATE_MACHINE"
#define EVENT_QUEUE_SIZE    16
#define EVENT_DATA_MAX      1024  /* TTS音声チャンクの最大バイト数 */

typedef struct {
    sm_event_t event;
    uint8_t    data[EVENT_DATA_MAX];
    size_t     data_len;
} sm_message_t;

static QueueHandle_t  s_event_queue  = NULL;
static TimerHandle_t  s_silence_timer = NULL;
static volatile sm_state_t s_state   = SM_STATE_IDLE;

/* --------------------------------------------------------
 * 状態遷移ヘルパー
 * -------------------------------------------------------- */
static void set_state(sm_state_t new_state) {
    const char *names[] = {"IDLE", "LISTENING", "VAD_SILENCE",
                            "PROCESSING", "SPEAKING"};
    ESP_LOGI(TAG, "状態遷移: %s → %s",
             names[s_state], names[new_state]);
    s_state = new_state;

    switch (new_state) {
        case SM_STATE_IDLE:
            lcd_set_state(LCD_STATE_IDLE);
            break;
        case SM_STATE_LISTENING:
        case SM_STATE_VAD_SILENCE:
            lcd_set_state(LCD_STATE_LISTENING);
            break;
        case SM_STATE_PROCESSING:
            lcd_set_state(LCD_STATE_PROCESSING);
            break;
        case SM_STATE_SPEAKING:
            lcd_set_state(LCD_STATE_SPEAKING);
            break;
    }
}

/* --------------------------------------------------------
 * 無音タイムアウトタイマーコールバック
 * -------------------------------------------------------- */
static void silence_timer_cb(TimerHandle_t timer) {
    sm_message_t msg = {.event = SM_EVENT_VAD_SILENCE_TIMEOUT, .data_len = 0};
    xQueueSend(s_event_queue, &msg, 0);
}

/* --------------------------------------------------------
 * メインイベントハンドラ
 * -------------------------------------------------------- */
static void handle_event(sm_event_t event, const uint8_t *data,
                          size_t data_len) {
    switch (s_state) {
        /* ---- IDLE: 待機中 ---- */
        case SM_STATE_IDLE:
            if (event == SM_EVENT_VAD_START) {
                set_state(SM_STATE_LISTENING);
            }
            break;

        /* ---- LISTENING: 録音中 ---- */
        case SM_STATE_LISTENING:
            if (event == SM_EVENT_VAD_END) {
                /* 無音検出: タイムアウトタイマー開始 */
                xTimerStart(s_silence_timer, 0);
                set_state(SM_STATE_VAD_SILENCE);
            }
            break;

        /* ---- VAD_SILENCE: タイムアウト待機 ---- */
        case SM_STATE_VAD_SILENCE:
            if (event == SM_EVENT_VAD_START) {
                /* 再び発話: タイマーキャンセルして録音継続 */
                xTimerStop(s_silence_timer, 0);
                set_state(SM_STATE_LISTENING);
            } else if (event == SM_EVENT_VAD_SILENCE_TIMEOUT) {
                /* タイムアウト: EOS送信してAI処理へ */
                ws_client_send_end_of_speech();
                set_state(SM_STATE_PROCESSING);
            }
            break;

        /* ---- PROCESSING: AI処理待ち ---- */
        case SM_STATE_PROCESSING:
            if (event == SM_EVENT_WS_TTS_CHUNK) {
                /* 最初のTTS音声チャンクが届いたら再生開始 */
                if (data && data_len > 0) {
                    audio_hal_play_bytes(data, data_len);
                }
                set_state(SM_STATE_SPEAKING);
            }
            break;

        /* ---- SPEAKING: TTS再生中 ---- */
        case SM_STATE_SPEAKING:
            if (event == SM_EVENT_VAD_START) {
                /* バージイン: 再生停止 → 割り込み送信 → 録音へ */
                audio_hal_stop_playback();
                ws_client_send_interrupt();
                set_state(SM_STATE_LISTENING);
            } else if (event == SM_EVENT_WS_TTS_CHUNK) {
                /* 後続のTTS音声チャンクを再生 */
                if (data && data_len > 0) {
                    audio_hal_play_bytes(data, data_len);
                }
            } else if (event == SM_EVENT_WS_TTS_END) {
                /* TTS完了: 待機へ */
                set_state(SM_STATE_IDLE);
            }
            break;
    }
}

/* --------------------------------------------------------
 * ステートマシンタスク (Core 0)
 * -------------------------------------------------------- */
static void state_machine_task(void *arg) {
    sm_message_t msg;
    while (true) {
        if (xQueueReceive(s_event_queue, &msg, portMAX_DELAY) == pdTRUE) {
            handle_event(msg.event, msg.data, msg.data_len);
        }
    }
}

/* --------------------------------------------------------
 * 公開API
 * -------------------------------------------------------- */

void state_machine_init(void) {
    s_event_queue = xQueueCreate(EVENT_QUEUE_SIZE, sizeof(sm_message_t));
    if (!s_event_queue) {
        ESP_LOGE(TAG, "イベントキュー確保失敗");
        return;
    }

    s_silence_timer = xTimerCreate(
        "silence_timer",
        pdMS_TO_TICKS(VAD_SILENCE_TIMEOUT_MS),
        pdFALSE,       /* 自動リロードなし (ワンショット) */
        NULL,
        silence_timer_cb);
    if (!s_silence_timer) {
        ESP_LOGE(TAG, "タイマー確保失敗");
        return;
    }

    xTaskCreatePinnedToCore(state_machine_task, "state_machine",
                             4096, NULL, 5, NULL, 0);
    ESP_LOGI(TAG, "ステートマシン初期化完了");
}

void state_machine_post_event(sm_event_t event, const void *data,
                               size_t data_len) {
    if (!s_event_queue) return;

    sm_message_t msg = {.event = event, .data_len = 0};

    if (data && data_len > 0) {
        size_t copy_len = data_len > EVENT_DATA_MAX ? EVENT_DATA_MAX : data_len;
        memcpy(msg.data, data, copy_len);
        msg.data_len = copy_len;
    }

    if (xQueueSend(s_event_queue, &msg, pdMS_TO_TICKS(50)) != pdTRUE) {
        ESP_LOGW(TAG, "イベントキュー満杯、イベント破棄 (%d)", event);
    }
}

sm_state_t state_machine_get_state(void) {
    return s_state;
}
