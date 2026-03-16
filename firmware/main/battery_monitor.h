#pragma once

#include <stdint.h>

/**
 * @brief バッテリー情報
 */
typedef struct {
    uint8_t level;        /* 0-100 (%) */
    uint8_t is_charging;  /* 0 or 1 */
    uint8_t is_usb_powered; /* 0 or 1 */
} battery_info_t;

/**
 * @brief バッテリー監視タスクを起動する
 *
 * PMICからバッテリー情報を30秒周期で読み取り、
 * WebSocket経由でサーバーへ送信する。
 * ws_client_init() の後に呼び出すこと。
 */
void battery_monitor_init(void);
