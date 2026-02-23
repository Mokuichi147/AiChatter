#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief クリーン音声受信コールバック (AEC + NS 処理済み)
 * @param samples  16kHz 16bit signed PCMサンプル配列
 * @param count    サンプル数
 */
typedef void (*audio_rx_callback_t)(const int16_t *samples, size_t count);

/**
 * @brief VAD (音声区間検出) イベントコールバック
 * @param voice_detected  true=発話検出, false=無音検出
 */
typedef void (*vad_event_callback_t)(bool voice_detected);

/**
 * @brief PMIC (M5PM1) を初期化し、LCD電源 (5V出力) を有効化する
 *
 * lcd_init() より前に呼び出すこと。
 * I2Cバスも同時に初期化する。
 */
void pmic_lcd_power_on(void);

/**
 * @brief 音声HALを初期化する
 *
 * I2S全二重モード (ES8311コーデック) と
 * ESP-SR AFE V2 (AEC+VAD) を初期化する。
 *
 * @param rx_cb   クリーン音声受信コールバック
 * @param vad_cb  VADイベントコールバック
 */
void audio_hal_init(audio_rx_callback_t rx_cb, vad_event_callback_t vad_cb);

/**
 * @brief TTS音声PCMデータをスピーカーに再生する
 *
 * データはAECリファレンス用リングバッファにも書き込む。
 *
 * @param data  16kHz 16bit signed PCMバイト列
 * @param len   バイト数
 */
void audio_hal_play_bytes(const uint8_t *data, size_t len);

/**
 * @brief スピーカー再生を即座に停止する (バージイン時に呼ぶ)
 */
void audio_hal_stop_playback(void);
