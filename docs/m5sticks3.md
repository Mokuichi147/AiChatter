# M5StickS3 連携ガイド

AiChatterサーバーにM5StickS3を接続して、ウェアラブル音声アシスタントとして利用するためのガイドです。

## アーキテクチャ

```
M5StickS3 (ESP32-S3)          PCサーバー (Python + uv)
┌───────────────────┐         ┌────────────────────────────────┐
│ ES8311 全二重音声 │◄──WS────│ FastAPI WebSocket               │
│ ESP-SR AFE        │  PCM    │ AudioPipeline                   │
│ (AEC + VAD)       │         │  ├── Qwen3-ASR (ASR / mlx)     │
│ ステートマシン     │         │  ├── Responses API互換LLM        │
│ LCD表示           │         │  ├── Qwen3-TTS (TTS / mlx)     │
│                   │         │  ├── ツール (記憶/通知/検索等)   │
│                   │         │  └── 話者識別 (WavLM-XVector)   │
└───────────────────┘         └────────────────────────────────┘
```

## 必要なもの

- M5StickS3 (ESP32-S3)
- ESP-IDF 5.3

## ファームウェアビルド

```bash
# ESP-IDF環境をロード
source ~/esp/esp-idf/export.sh

cd firmware

# WiFi/サーバーIP設定
nano main/config.h

# WS_SERVER_URI は /ws?device=m5 を含める
# 例: ws://192.168.11.52:8765/ws?device=m5

idf.py set-target esp32s3
idf.py build
```

## WebSocketプロトコル（7バイトバイナリヘッダー）

| type  | 方向           | 意味              |
|-------|----------------|-------------------|
| 0x01  | ESP32 → Server | 音声チャンク       |
| 0x11  | ESP32 → Server | 発話終了(EOS)      |
| 0x12  | ESP32 → Server | バージイン割り込み  |
| 0x13  | ESP32 → Server | ボタン押下         |
| 0x02  | Server → ESP32 | TTS音声チャンク    |
| 0x03  | Server → ESP32 | TTS終了            |
| 0x04  | Server → ESP32 | スリープ指示       |
| 0x05  | Server → ESP32 | ウェイク指示       |
| 0x20  | Server → ESP32 | テキスト表示       |
| 0x21  | Server → ESP32 | 画像ブロック表示   |

ヘッダー構造: `[type:1][seq:2][payload_len:4]` (ビッグエンディアン)

## 音声フォーマット

- サンプルレート: 16kHz
- ビット深度: 16bit signed PCM
- チャンネル: mono
- チャンクサイズ: 512サンプル (32ms)

## ハードウェアピン（M5StickS3）

| 機能 | GPIO |
|------|------|
| I2S MCLK | 18 |
| I2S BCLK | 17 |
| I2S WS | 15 |
| I2S DOUT | 14 |
| I2S DIN | 16 |
| I2C SDA (内部) | 47 |
| I2C SCL (内部) | 48 |
| I2C SDA (Grove) | 9 |
| I2C SCL (Grove) | 10 |
| ボタンA (フロント) | 11 |
| ボタンB (サイド/電源) | 12 |
| LCD MOSI | 39 |
| LCD SCLK | 40 |
| LCD CS | 41 |
| LCD DC | 45 |
| LCD RST | 21 |
| LCD BL | 38 |
| IR送信 | 46 |
| IR受信 | 42 |

## LCD状態表示

| 色     | 状態       |
|--------|-----------|
| 黒     | IDLE       |
| 青     | LISTENING  |
| 黄     | PROCESSING |
| 緑     | SPEAKING   |
