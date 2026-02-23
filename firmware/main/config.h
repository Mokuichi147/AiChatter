#pragma once

// ============================================================
// WiFi設定 (ここを書き換えてください)
// ============================================================
#define WIFI_SSID       "Buffalo-2.4GHz-7D60"
#define WIFI_PASSWORD   "xf5myfwsd5prb"

// ============================================================
// サーバー設定 (PCのIPアドレスを設定してください)
// ============================================================
#define WS_SERVER_URI   "ws://192.168.11.19:8765/ws"

// ============================================================
// ハードウェア設定 (M5StickS3固定値 - 変更不要)
// ============================================================

// I2S (ES8311 音声コーデック)
#define I2S_MCLK_GPIO   18
#define I2S_BCLK_GPIO   17
#define I2S_WS_GPIO     15
#define I2S_DOUT_GPIO   14   // DAC出力 (スピーカーへ)
#define I2S_DIN_GPIO    16   // ADC入力 (マイクから)

// I2C (ES8311 制御)
#define I2C_SDA_GPIO    47
#define I2C_SCL_GPIO    48
#define ES8311_I2C_ADDR 0x18

// ボタン (A=フロント, B=サイド/電源)
#define BUTTON_A_GPIO   11
#define BUTTON_B_GPIO   12

// PMIC (M5PM1, I2C 0x6E) - LCD電源・ES8311・AW8737制御
// M5PM1 有効レジスタ: 0x06(5V出力), 0x0C(電源OFF), 0x10(GPIO方向),
//                    0x11(GPIO出力値), 0x12(GPIO状態), 0x16(GPIO機能選択)
// GPIO3 bit3: ES8311/AW8737電源 (0x10 bit3=1:出力, 0x11 bit3=1:HIGH)
#define PMIC_I2C_ADDR       0x6E

// LCD (ST7789 135x240)
#define LCD_MOSI_GPIO   39
#define LCD_SCLK_GPIO   40
#define LCD_CS_GPIO     41
#define LCD_DC_GPIO     45
#define LCD_RST_GPIO    21
#define LCD_BL_GPIO     38
#define LCD_WIDTH       135
#define LCD_HEIGHT      240

// ============================================================
// 音声設定
// ============================================================
#define AUDIO_SAMPLE_RATE   16000
#define AUDIO_CHUNK_SAMPLES 512      // 32ms per chunk
#define AUDIO_CHUNK_BYTES   (AUDIO_CHUNK_SAMPLES * 2)  // 16bit = 2 bytes/sample

// ============================================================
// VAD設定
// ============================================================
#define VAD_SILENCE_TIMEOUT_MS  1500  // 1.5秒無音でEOS送信
