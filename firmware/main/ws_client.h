#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief TTS音声チャンク受信コールバック
 * @param data  PCMデータ (16kHz, 16bit signed, mono)
 * @param len   バイト数
 */
typedef void (*tts_audio_callback_t)(const uint8_t *data, size_t len);

/**
 * @brief TTS終了通知コールバック
 */
typedef void (*tts_end_callback_t)(void);

/**
 * @brief WebSocketクライアントを初期化して接続する
 * @param uri       ws://host:port/path 形式のURI
 * @param tts_cb    TTS音声チャンク受信時コールバック
 * @param end_cb    TTS終了時コールバック
 */
void ws_client_init(const char *uri, tts_audio_callback_t tts_cb,
                    tts_end_callback_t end_cb);

/**
 * @brief 音声チャンクをサーバーへ送信 (type=0x01)
 * @param samples   16bit signed PCMサンプル配列
 * @param count     サンプル数
 */
void ws_client_send_audio(const int16_t *samples, size_t count);

/**
 * @brief 発話終了 (EOS) をサーバーへ送信 (type=0x11)
 */
void ws_client_send_end_of_speech(void);

/**
 * @brief バージイン割り込みをサーバーへ送信 (type=0x12)
 */
void ws_client_send_interrupt(void);

/**
 * @brief ボタン押下をサーバーへ送信 (type=0x13)
 */
void ws_client_send_button(void);

/**
 * @brief バッテリー情報をサーバーへ送信 (type=0x14)
 * @param level       バッテリー残量 (0-100)
 * @param charging    充電中フラグ (0 or 1)
 * @param usb_powered USB給電フラグ (0 or 1)
 */
void ws_client_send_battery_info(uint8_t level, uint8_t charging,
                                 uint8_t usb_powered);

/**
 * @brief WebSocket接続状態を返す
 */
bool ws_client_is_connected(void);

/**
 * @brief オフライン音声バッファをクリアする
 */
void ws_client_clear_audio_buffer(void);
