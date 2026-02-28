#pragma once
#include <stdarg.h>

/**
 * @brief LCD表示状態
 * 色で現在の動作状態を示す
 */
typedef enum {
    LCD_STATE_IDLE       = 0,  /**< 待機中 (黒) */
    LCD_STATE_LISTENING  = 1,  /**< 録音中 (青) */
    LCD_STATE_PROCESSING = 2,  /**< AI処理中 (黄) */
    LCD_STATE_SPEAKING   = 3,  /**< 音声再生中 (緑) */
    LCD_STATE_SLEEP      = 4,  /**< スリープ中 (バックライト消灯) */
} lcd_state_t;

/**
 * @brief LCDを初期化する (SPI2 + ST7789 + LEDCバックライト)
 */
void lcd_init(void);

/**
 * @brief LCD状態表示を更新する
 * @param state 表示する状態
 */
void lcd_set_state(lcd_state_t state);

/**
 * @brief LCDにログ行を追加する (スクロール表示)
 * @param fmt printf形式フォーマット文字列
 */
void lcd_log(const char *fmt, ...);

/**
 * @brief バックライト消灯 (スリープ時)
 */
void lcd_sleep(void);

/**
 * @brief バックライト復帰 (ウェイク時)
 */
void lcd_wake(void);
