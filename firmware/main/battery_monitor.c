#include "battery_monitor.h"

#include "audio_hal.h"
#include "config.h"
#include "ws_client.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "BATTERY"

/* バッテリー監視間隔 (30秒) */
#define BATTERY_CHECK_INTERVAL_MS  30000

/* バッテリー電圧 → 残量変換 (リチウムイオン 3.7V)
 * 放電カーブの近似: 空=3300mV, 満充電=4200mV */
#define VBAT_MIN_MV  3300
#define VBAT_MAX_MV  4200

/**
 * @brief バッテリー電圧(mV)から残量(%)に変換する
 */
static uint8_t voltage_to_percent(uint16_t mv) {
    if (mv >= VBAT_MAX_MV) return 100;
    if (mv <= VBAT_MIN_MV) return 0;
    return (uint8_t)(((uint32_t)(mv - VBAT_MIN_MV) * 100) /
                     (VBAT_MAX_MV - VBAT_MIN_MV));
}

/**
 * @brief バッテリー情報を読み取る
 * @return true=読み取り成功, false=失敗
 */
static bool read_battery_info(battery_info_t *info) {
    uint8_t vbat_buf[2] = {0};
    uint8_t gpio_in = 0xFF;
    uint8_t pwr_src = 0xFF;

    /* バッテリー電圧読み取り (0x22-0x23) */
    if (pmic_read_reg(0x22, vbat_buf, 2) != 0) {
        ESP_LOGW(TAG, "VBAT読み取り失敗");
        return false;
    }

    /* GPIO入力状態読み取り (0x12) */
    if (pmic_read_reg(0x12, &gpio_in, 1) != 0) {
        ESP_LOGW(TAG, "GPIO_IN読み取り失敗");
        return false;
    }

    /* 電源ソース読み取り (0x04) */
    if (pmic_read_reg(0x04, &pwr_src, 1) != 0) {
        ESP_LOGW(TAG, "PWR_SRC読み取り失敗");
        return false;
    }

    uint16_t vbat_mv = ((uint16_t)vbat_buf[1] << 8) | vbat_buf[0];
    info->level = voltage_to_percent(vbat_mv);
    /* GPIO0 bit0=0 → 充電中, bit0=1 → 放電中 */
    info->is_charging = (gpio_in & 0x01) == 0 ? 1 : 0;
    /* 電源ソース [2:0]: 0=5VIN(USB), 1=5VINOUT, 2=BAT */
    info->is_usb_powered = ((pwr_src & 0x07) <= 1) ? 1 : 0;

    ESP_LOGI(TAG, "VBAT=%umV (%u%%) charging=%u usb=%u (gpio_in=0x%02X pwr_src=0x%02X)",
             vbat_mv, info->level, info->is_charging, info->is_usb_powered,
             gpio_in, pwr_src);

    return true;
}

/* --------------------------------------------------------
 * バッテリー監視タスク (Core 0, 優先度 2)
 * -------------------------------------------------------- */
static void battery_monitor_task(void *arg) {
    /* 起動直後は他の初期化を待つ */
    vTaskDelay(pdMS_TO_TICKS(5000));

    while (true) {
        battery_info_t info = {0};
        if (read_battery_info(&info)) {
            ws_client_send_battery_info(info.level, info.is_charging,
                                        info.is_usb_powered);
        }
        vTaskDelay(pdMS_TO_TICKS(BATTERY_CHECK_INTERVAL_MS));
    }
}

void battery_monitor_init(void) {
    xTaskCreatePinnedToCore(battery_monitor_task, "bat_mon",
                             3072, NULL, 2, NULL, 0);
    ESP_LOGI(TAG, "バッテリー監視タスク起動 (%d秒周期)",
             BATTERY_CHECK_INTERVAL_MS / 1000);
}
