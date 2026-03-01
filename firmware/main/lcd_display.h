#pragma once
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

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
 * @brief 画面にテキストを表示する
 * @param text 表示する文字列
 * @param size 文字サイズ倍率 (1-4)
 * @param x 左上X座標
 * @param y 左上Y座標
 * @param clear trueなら本文領域をクリアしてから描画
 */
void lcd_show_text(const char *text, uint8_t size, uint16_t x, uint16_t y, bool clear);

/**
 * @brief RGB565画像ブロックを描画する
 * @param x 左上X座標
 * @param y 左上Y座標
 * @param width 幅(px)
 * @param height 高さ(px)
 * @param data RGB565ビッグエンディアンデータ
 * @param data_len dataのバイト数
 */
void lcd_draw_rgb565(uint16_t x, uint16_t y, uint16_t width, uint16_t height,
                     const uint8_t *data, size_t data_len);

/**
 * @brief バックライト消灯 (スリープ時)
 */
void lcd_sleep(void);

/**
 * @brief バックライト復帰 (ウェイク時)
 */
void lcd_wake(void);
