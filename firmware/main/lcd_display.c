#include "lcd_display.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/spi_master.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#define TAG "LCD"

/* ST7789 コマンド */
#define ST7789_SWRESET  0x01
#define ST7789_SLPOUT   0x11
#define ST7789_COLMOD   0x3A
#define ST7789_MADCTL   0x36
#define ST7789_INVON    0x21
#define ST7789_DISPON   0x29
#define ST7789_CASET    0x2A
#define ST7789_RASET    0x2B
#define ST7789_RAMWR    0x2C

/* M5StickS3 LCD オフセット (135x240 ST7789) */
#define LCD_X_OFFSET    52
#define LCD_Y_OFFSET    40

/* RGB565カラー */
#define COLOR_BLACK     0x0000
#define COLOR_BLUE      0x001F
#define COLOR_YELLOW    0xFFE0
#define COLOR_GREEN     0x07E0
#define COLOR_WHITE     0xFFFF
#define COLOR_RED       0xF800
#define COLOR_GRAY      0x4208

/* ログテキスト設定 */
#define FONT_W          6    /* フォント幅(px) */
#define FONT_H          8    /* フォント高さ(px) */
#define LOG_COLS        (LCD_WIDTH  / FONT_W)   /* 1行の文字数: 22 */
#define LOG_STATUS_H    10   /* 上部ステータスバー高さ(px) */
#define LOG_ROWS        ((LCD_HEIGHT - LOG_STATUS_H) / FONT_H)  /* ログ行数: 28 */

static spi_device_handle_t s_spi = NULL;
static SemaphoreHandle_t   s_lcd_mutex = NULL;

/* ログバッファ */
static char  s_log_lines[LOG_ROWS][LOG_COLS + 1];
static int   s_log_count = 0;   /* 総追加行数 */

/* ステータスバー文字列 */
static char s_status_str[LOG_COLS + 1] = "AiChatter";
static uint16_t s_status_color = COLOR_BLACK;

/* ========== 6x8 ビットマップフォント (ASCII 0x20-0x7E) ==========
 * 各文字は 6バイト(6列) x 8行。ビット[0]=最上行、ビット[7]=最下行。
 * 幅6px: 上位6ビットを使用 (bit7=左端)。
 * Public domain 6x8 font (condensed version of Terminus/System fonts).
 */
static const uint8_t FONT6x8[95][6] = {
 {0x00,0x00,0x00,0x00,0x00,0x00}, /* 20 space */
 {0x00,0x00,0x5F,0x00,0x00,0x00}, /* 21 ! */
 {0x00,0x07,0x00,0x07,0x00,0x00}, /* 22 " */
 {0x14,0x7F,0x14,0x7F,0x14,0x00}, /* 23 # */
 {0x24,0x2A,0x7F,0x2A,0x12,0x00}, /* 24 $ */
 {0x23,0x13,0x08,0x64,0x62,0x00}, /* 25 % */
 {0x36,0x49,0x55,0x22,0x50,0x00}, /* 26 & */
 {0x00,0x05,0x03,0x00,0x00,0x00}, /* 27 ' */
 {0x00,0x1C,0x22,0x41,0x00,0x00}, /* 28 ( */
 {0x00,0x41,0x22,0x1C,0x00,0x00}, /* 29 ) */
 {0x08,0x2A,0x1C,0x2A,0x08,0x00}, /* 2A * */
 {0x08,0x08,0x3E,0x08,0x08,0x00}, /* 2B + */
 {0x00,0x50,0x30,0x00,0x00,0x00}, /* 2C , */
 {0x08,0x08,0x08,0x08,0x08,0x00}, /* 2D - */
 {0x00,0x60,0x60,0x00,0x00,0x00}, /* 2E . */
 {0x20,0x10,0x08,0x04,0x02,0x00}, /* 2F / */
 {0x3E,0x51,0x49,0x45,0x3E,0x00}, /* 30 0 */
 {0x00,0x42,0x7F,0x40,0x00,0x00}, /* 31 1 */
 {0x42,0x61,0x51,0x49,0x46,0x00}, /* 32 2 */
 {0x21,0x41,0x45,0x4B,0x31,0x00}, /* 33 3 */
 {0x18,0x14,0x12,0x7F,0x10,0x00}, /* 34 4 */
 {0x27,0x45,0x45,0x45,0x39,0x00}, /* 35 5 */
 {0x3C,0x4A,0x49,0x49,0x30,0x00}, /* 36 6 */
 {0x01,0x71,0x09,0x05,0x03,0x00}, /* 37 7 */
 {0x36,0x49,0x49,0x49,0x36,0x00}, /* 38 8 */
 {0x06,0x49,0x49,0x29,0x1E,0x00}, /* 39 9 */
 {0x00,0x36,0x36,0x00,0x00,0x00}, /* 3A : */
 {0x00,0x56,0x36,0x00,0x00,0x00}, /* 3B ; */
 {0x00,0x08,0x14,0x22,0x41,0x00}, /* 3C < */
 {0x14,0x14,0x14,0x14,0x14,0x00}, /* 3D = */
 {0x41,0x22,0x14,0x08,0x00,0x00}, /* 3E > */
 {0x02,0x01,0x51,0x09,0x06,0x00}, /* 3F ? */
 {0x32,0x49,0x79,0x41,0x3E,0x00}, /* 40 @ */
 {0x7E,0x11,0x11,0x11,0x7E,0x00}, /* 41 A */
 {0x7F,0x49,0x49,0x49,0x36,0x00}, /* 42 B */
 {0x3E,0x41,0x41,0x41,0x22,0x00}, /* 43 C */
 {0x7F,0x41,0x41,0x22,0x1C,0x00}, /* 44 D */
 {0x7F,0x49,0x49,0x49,0x41,0x00}, /* 45 E */
 {0x7F,0x09,0x09,0x09,0x01,0x00}, /* 46 F */
 {0x3E,0x41,0x49,0x49,0x7A,0x00}, /* 47 G */
 {0x7F,0x08,0x08,0x08,0x7F,0x00}, /* 48 H */
 {0x00,0x41,0x7F,0x41,0x00,0x00}, /* 49 I */
 {0x20,0x40,0x41,0x3F,0x01,0x00}, /* 4A J */
 {0x7F,0x08,0x14,0x22,0x41,0x00}, /* 4B K */
 {0x7F,0x40,0x40,0x40,0x40,0x00}, /* 4C L */
 {0x7F,0x02,0x04,0x02,0x7F,0x00}, /* 4D M */
 {0x7F,0x04,0x08,0x10,0x7F,0x00}, /* 4E N */
 {0x3E,0x41,0x41,0x41,0x3E,0x00}, /* 4F O */
 {0x7F,0x09,0x09,0x09,0x06,0x00}, /* 50 P */
 {0x3E,0x41,0x51,0x21,0x5E,0x00}, /* 51 Q */
 {0x7F,0x09,0x19,0x29,0x46,0x00}, /* 52 R */
 {0x46,0x49,0x49,0x49,0x31,0x00}, /* 53 S */
 {0x01,0x01,0x7F,0x01,0x01,0x00}, /* 54 T */
 {0x3F,0x40,0x40,0x40,0x3F,0x00}, /* 55 U */
 {0x1F,0x20,0x40,0x20,0x1F,0x00}, /* 56 V */
 {0x3F,0x40,0x38,0x40,0x3F,0x00}, /* 57 W */
 {0x63,0x14,0x08,0x14,0x63,0x00}, /* 58 X */
 {0x07,0x08,0x70,0x08,0x07,0x00}, /* 59 Y */
 {0x61,0x51,0x49,0x45,0x43,0x00}, /* 5A Z */
 {0x00,0x7F,0x41,0x41,0x00,0x00}, /* 5B [ */
 {0x02,0x04,0x08,0x10,0x20,0x00}, /* 5C \ */
 {0x00,0x41,0x41,0x7F,0x00,0x00}, /* 5D ] */
 {0x04,0x02,0x01,0x02,0x04,0x00}, /* 5E ^ */
 {0x40,0x40,0x40,0x40,0x40,0x00}, /* 5F _ */
 {0x00,0x01,0x02,0x04,0x00,0x00}, /* 60 ` */
 {0x20,0x54,0x54,0x54,0x78,0x00}, /* 61 a */
 {0x7F,0x48,0x44,0x44,0x38,0x00}, /* 62 b */
 {0x38,0x44,0x44,0x44,0x20,0x00}, /* 63 c */
 {0x38,0x44,0x44,0x48,0x7F,0x00}, /* 64 d */
 {0x38,0x54,0x54,0x54,0x18,0x00}, /* 65 e */
 {0x08,0x7E,0x09,0x01,0x02,0x00}, /* 66 f */
 {0x0C,0x52,0x52,0x52,0x3E,0x00}, /* 67 g */
 {0x7F,0x08,0x04,0x04,0x78,0x00}, /* 68 h */
 {0x00,0x44,0x7D,0x40,0x00,0x00}, /* 69 i */
 {0x20,0x40,0x44,0x3D,0x00,0x00}, /* 6A j */
 {0x7F,0x10,0x28,0x44,0x00,0x00}, /* 6B k */
 {0x00,0x41,0x7F,0x40,0x00,0x00}, /* 6C l */
 {0x7C,0x04,0x18,0x04,0x78,0x00}, /* 6D m */
 {0x7C,0x08,0x04,0x04,0x78,0x00}, /* 6E n */
 {0x38,0x44,0x44,0x44,0x38,0x00}, /* 6F o */
 {0x7C,0x14,0x14,0x14,0x08,0x00}, /* 70 p */
 {0x08,0x14,0x14,0x18,0x7C,0x00}, /* 71 q */
 {0x7C,0x08,0x04,0x04,0x08,0x00}, /* 72 r */
 {0x48,0x54,0x54,0x54,0x20,0x00}, /* 73 s */
 {0x04,0x3F,0x44,0x40,0x20,0x00}, /* 74 t */
 {0x3C,0x40,0x40,0x20,0x7C,0x00}, /* 75 u */
 {0x1C,0x20,0x40,0x20,0x1C,0x00}, /* 76 v */
 {0x3C,0x40,0x30,0x40,0x3C,0x00}, /* 77 w */
 {0x44,0x28,0x10,0x28,0x44,0x00}, /* 78 x */
 {0x0C,0x50,0x50,0x50,0x3C,0x00}, /* 79 y */
 {0x44,0x64,0x54,0x4C,0x44,0x00}, /* 7A z */
 {0x00,0x08,0x36,0x41,0x00,0x00}, /* 7B { */
 {0x00,0x00,0x7F,0x00,0x00,0x00}, /* 7C | */
 {0x00,0x41,0x36,0x08,0x00,0x00}, /* 7D } */
 {0x08,0x04,0x08,0x10,0x08,0x00}, /* 7E ~ */
};

static void lcd_cmd(uint8_t cmd) {
    gpio_set_level(LCD_DC_GPIO, 0);
    spi_transaction_t t = {
        .length = 8,
        .tx_buffer = &cmd,
        .flags = 0,
    };
    spi_device_polling_transmit(s_spi, &t);
}

static void lcd_data(const uint8_t *data, size_t len) {
    if (len == 0) return;
    gpio_set_level(LCD_DC_GPIO, 1);
    spi_transaction_t t = {
        .length = len * 8,
        .tx_buffer = data,
        .flags = 0,
    };
    spi_device_polling_transmit(s_spi, &t);
}

static void lcd_data_byte(uint8_t b) {
    lcd_data(&b, 1);
}

static void lcd_set_window(uint16_t x0, uint16_t y0, uint16_t x1,
                            uint16_t y1) {
    uint8_t buf[4];

    lcd_cmd(ST7789_CASET);
    uint16_t xa = x0 + LCD_X_OFFSET;
    uint16_t xb = x1 + LCD_X_OFFSET;
    buf[0] = xa >> 8;
    buf[1] = xa & 0xFF;
    buf[2] = xb >> 8;
    buf[3] = xb & 0xFF;
    lcd_data(buf, 4);

    lcd_cmd(ST7789_RASET);
    uint16_t ya = y0 + LCD_Y_OFFSET;
    uint16_t yb = y1 + LCD_Y_OFFSET;
    buf[0] = ya >> 8;
    buf[1] = ya & 0xFF;
    buf[2] = yb >> 8;
    buf[3] = yb & 0xFF;
    lcd_data(buf, 4);

    lcd_cmd(ST7789_RAMWR);
}

/* 画面全体を単色で塗りつぶす */
static void lcd_fill(uint16_t color) {
    lcd_set_window(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1);

    uint8_t hi = color >> 8;
    uint8_t lo = color & 0xFF;

    /* ライン単位で送信 (スタックオーバーフロー防止) */
    static uint8_t line_buf[LCD_WIDTH * 2];
    for (int i = 0; i < LCD_WIDTH; i++) {
        line_buf[i * 2] = hi;
        line_buf[i * 2 + 1] = lo;
    }

    gpio_set_level(LCD_DC_GPIO, 1);
    for (int y = 0; y < LCD_HEIGHT; y++) {
        spi_transaction_t t = {
            .length = LCD_WIDTH * 16,
            .tx_buffer = line_buf,
        };
        spi_device_polling_transmit(s_spi, &t);
    }
}

/* --------------------------------------------------------
 * 1文字描画 (6x8 px, fg/bg色)
 * -------------------------------------------------------- */
static void lcd_draw_char(int x, int y, char c, uint16_t fg, uint16_t bg) {
    if (c < 0x20 || c > 0x7E) c = '?';
    const uint8_t *glyph = FONT6x8[(uint8_t)c - 0x20];

    static uint8_t buf[FONT_W * FONT_H * 2];  /* 6*8*2=96bytes */
    int idx = 0;
    for (int row = 0; row < FONT_H; row++) {
        for (int col = 0; col < FONT_W; col++) {
            /* bit0=最上行 (row=0のとき bit0 をチェック) */
            uint16_t color = (glyph[col] & (1 << row)) ? fg : bg;
            buf[idx++] = color >> 8;
            buf[idx++] = color & 0xFF;
        }
    }
    lcd_set_window(x, y, x + FONT_W - 1, y + FONT_H - 1);
    lcd_data(buf, sizeof(buf));
}

/* 文字列を指定座標に描画 */
static void lcd_draw_string(int x, int y, const char *str, uint16_t fg, uint16_t bg) {
    while (*str && x + FONT_W <= LCD_WIDTH) {
        lcd_draw_char(x, y, *str++, fg, bg);
        x += FONT_W;
    }
    /* 残り部分をbgで塗りつぶし */
    while (x + FONT_W <= LCD_WIDTH) {
        lcd_draw_char(x, y, ' ', fg, bg);
        x += FONT_W;
    }
}

/* ステータスバーを描画 (上部 LOG_STATUS_H px) */
static void lcd_redraw_status(void) {
    /* 短い行で埋める */
    static char padded[LOG_COLS + 1];
    snprintf(padded, sizeof(padded), "%-*s", LOG_COLS, s_status_str);
    /* ステータスバーを塗りつぶしながら文字描画 */
    for (int i = 0; i < LOG_COLS && i < (int)strlen(padded); i++) {
        lcd_draw_char(i * FONT_W, 0, padded[i], COLOR_BLACK, s_status_color);
    }
}

/* ログ領域を全再描画 */
static void lcd_redraw_log(void) {
    int total = s_log_count < LOG_ROWS ? s_log_count : LOG_ROWS;
    int start = s_log_count > LOG_ROWS ? (s_log_count % LOG_ROWS) : 0;

    for (int row = 0; row < LOG_ROWS; row++) {
        int y = LOG_STATUS_H + row * FONT_H;
        if (row < total) {
            int idx = (start + row) % LOG_ROWS;
            lcd_draw_string(0, y, s_log_lines[idx], COLOR_WHITE, COLOR_BLACK);
        } else {
            /* 空行を黒で塗りつぶし */
            lcd_draw_string(0, y, "", COLOR_WHITE, COLOR_BLACK);
        }
    }
}

void lcd_init(void) {
    /* GPIO設定 */
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LCD_DC_GPIO) | (1ULL << LCD_RST_GPIO) |
                        (1ULL << LCD_CS_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);

    /* SPI バス初期化 */
    spi_bus_config_t buscfg = {
        .mosi_io_num = LCD_MOSI_GPIO,
        .miso_io_num = -1,
        .sclk_io_num = LCD_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LCD_WIDTH * LCD_HEIGHT * 2 + 8,
    };
    spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO);

    /* LCDデバイス登録 */
    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = 40 * 1000 * 1000,  /* 40MHz */
        .mode = 0,
        .spics_io_num = LCD_CS_GPIO,
        .queue_size = 7,
    };
    spi_bus_add_device(SPI2_HOST, &devcfg, &s_spi);

    /* ハードウェアリセット */
    gpio_set_level(LCD_RST_GPIO, 0);
    vTaskDelay(pdMS_TO_TICKS(100));
    gpio_set_level(LCD_RST_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(100));

    /* ST7789 初期化シーケンス */
    lcd_cmd(ST7789_SWRESET);
    vTaskDelay(pdMS_TO_TICKS(150));
    lcd_cmd(ST7789_SLPOUT);
    vTaskDelay(pdMS_TO_TICKS(10));
    lcd_cmd(ST7789_COLMOD);
    lcd_data_byte(0x55);  /* 16bit/pixel (RGB565) */
    lcd_cmd(ST7789_MADCTL);
    lcd_data_byte(0x00);  /* 通常方向 */
    lcd_cmd(ST7789_INVON);
    vTaskDelay(pdMS_TO_TICKS(10));
    lcd_cmd(ST7789_DISPON);
    vTaskDelay(pdMS_TO_TICKS(10));

    /* バックライト (LEDC PWM) */
    ledc_timer_config_t ledc_timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .freq_hz = 5000,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&ledc_timer);

    ledc_channel_config_t ledc_channel = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_0,
        .timer_sel = LEDC_TIMER_0,
        .intr_type = LEDC_INTR_DISABLE,
        .gpio_num = LCD_BL_GPIO,
        .duty = 200,  /* 約78% */
        .hpoint = 0,
    };
    ledc_channel_config(&ledc_channel);

    s_lcd_mutex = xSemaphoreCreateMutex();
    memset(s_log_lines, 0, sizeof(s_log_lines));
    s_log_count = 0;

    /* 画面全体を黒で初期化 */
    lcd_fill(COLOR_BLACK);

    /* ステータスバー初期表示 */
    s_status_color = COLOR_GRAY;
    snprintf(s_status_str, sizeof(s_status_str), "AiChatter");
    lcd_redraw_status();

    ESP_LOGI(TAG, "LCD初期化完了 (%dx%d)", LCD_WIDTH, LCD_HEIGHT);
}

void lcd_set_state(lcd_state_t state) {
    uint16_t color;
    const char *name;

    switch (state) {
        case LCD_STATE_IDLE:
            color = COLOR_GRAY;
            name = "IDLE";
            break;
        case LCD_STATE_LISTENING:
            color = COLOR_BLUE;
            name = "LISTENING";
            break;
        case LCD_STATE_PROCESSING:
            color = COLOR_YELLOW;
            name = "PROCESSING";
            break;
        case LCD_STATE_SPEAKING:
            color = COLOR_GREEN;
            name = "SPEAKING";
            break;
        default:
            color = COLOR_GRAY;
            name = "UNKNOWN";
            break;
    }

    if (s_lcd_mutex && xSemaphoreTake(s_lcd_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        s_status_color = color;
        snprintf(s_status_str, sizeof(s_status_str), "%-*s", LOG_COLS, name);
        lcd_redraw_status();
        xSemaphoreGive(s_lcd_mutex);
    }
    ESP_LOGD(TAG, "LCD状態: %s", name);
}

void lcd_log(const char *fmt, ...) {
    if (!s_lcd_mutex) return;

    char line[LOG_COLS + 1];
    va_list args;
    va_start(args, fmt);
    vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);

    /* ログバッファに追加 */
    int idx = s_log_count % LOG_ROWS;
    strncpy(s_log_lines[idx], line, LOG_COLS);
    s_log_lines[idx][LOG_COLS] = '\0';
    s_log_count++;

    if (xSemaphoreTake(s_lcd_mutex, pdMS_TO_TICKS(200)) == pdTRUE) {
        lcd_redraw_log();
        xSemaphoreGive(s_lcd_mutex);
    }
}
