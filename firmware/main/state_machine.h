#pragma once

#include <stddef.h>
#include <stdint.h>

/**
 * @brief システム動作状態
 */
typedef enum {
    SM_STATE_IDLE        = 0, /**< 待機中 */
    SM_STATE_LISTENING   = 1, /**< 発話録音中 */
    SM_STATE_VAD_SILENCE = 2, /**< 無音検出後タイムアウト待機中 */
    SM_STATE_PROCESSING  = 3, /**< AIパイプライン処理中 */
    SM_STATE_SPEAKING    = 4, /**< TTS音声再生中 */
} sm_state_t;

/**
 * @brief イベント種別
 */
typedef enum {
    SM_EVENT_VAD_START,           /**< 発話開始 (VAD検出) */
    SM_EVENT_VAD_END,             /**< 発話終了 (無音検出) */
    SM_EVENT_VAD_SILENCE_TIMEOUT, /**< 無音タイムアウト → EOS送信 */
    SM_EVENT_WS_TTS_CHUNK,        /**< サーバーからTTS音声受信 */
    SM_EVENT_WS_TTS_END,          /**< サーバーからTTS終了受信 */
} sm_event_t;

/**
 * @brief ステートマシンを初期化してタスクを起動する
 */
void state_machine_init(void);

/**
 * @brief イベントをステートマシンキューに投入する (ISR/タスクから安全)
 * @param event     イベント種別
 * @param data      付随データ (NULL可)
 * @param data_len  データバイト数
 */
void state_machine_post_event(sm_event_t event, const void *data,
                               size_t data_len);

/**
 * @brief 現在の状態を返す
 */
sm_state_t state_machine_get_state(void);
