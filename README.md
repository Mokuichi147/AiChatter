# AiChatter - M5StickS3 常時会話AI音声アシスタント

M5StickS3を応答中でも割り込み可能（バージイン対応）な常時会話型AI音声アシスタントにするプロジェクト。
すべてのAI処理はローカル実行（クラウドAPI不使用）。

## アーキテクチャ

```
M5StickS3 (ESP32-S3)          PCサーバー (Python + uv)
┌───────────────────┐         ┌────────────────────────────┐
│ ES8311 全二重音声 │◄──WS────│ FastAPI WebSocket           │
│ ESP-SR AFE        │  PCM    │ AudioPipeline               │
│ (AEC + VAD)       │         │  ├── faster-whisper (ASR)   │
│ ステートマシン     │         │  ├── Ollama (LLM)           │
│ LCD表示           │         │  └── piper-tts (TTS)        │
└───────────────────┘         └────────────────────────────┘
```

## セットアップ

### 必要なもの
- M5StickS3 (ESP32-S3)
- Python 3.10+ / uv
- ESP-IDF 5.3
- Ollama (granite4:micro-h または qwen2.5:7b など)

### Stage 0: 開発環境セットアップ

#### ESP-IDF 5.3 インストール
```bash
mkdir ~/esp && cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && git checkout v5.3
./install.sh esp32s3
source export.sh   # 毎回必要（.zshrcに追加推奨）
```

#### Pythonサーバー依存関係インストール
```bash
cd AiChatter/server
uv sync
```

#### piper-tts 日本語モデルのダウンロード
```bash
cd server
mkdir -p models
uv run python -c "
import urllib.request, os
base = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ja/ja_JP/kokoro/medium'
urllib.request.urlretrieve(base + '/ja_JP-kokoro-medium.onnx', 'models/tts.onnx')
urllib.request.urlretrieve(base + '/ja_JP-kokoro-medium.onnx.json', 'models/tts.onnx.json')
print('Done')
"
```

### Stage 1: サーバー起動

```bash
cd server
cp .env.example .env
# .envを編集してモデルパス等を設定
uv run python main.py
```

### Stage 2: ファームウェアビルド

```bash
# ESP-IDF環境をロード済みで:
cd firmware

# WiFi/サーバーIP設定 (main/config.hを編集)
nano main/config.h

idf.py set-target esp32s3
idf.py -p /dev/cu.usbmodem1101 flash monitor
```

## WebSocketプロトコル（7バイトバイナリヘッダー）

| type  | 方向           | 意味              |
|-------|----------------|-------------------|
| 0x01  | ESP32 → Server | 音声チャンク       |
| 0x11  | ESP32 → Server | 発話終了(EOS)      |
| 0x12  | ESP32 → Server | バージイン割り込み  |
| 0x02  | Server → ESP32 | TTS音声チャンク    |
| 0x03  | Server → ESP32 | TTS終了            |
| 0x20  | Server → ESP32 | テキスト表示       |
| 0x21  | Server → ESP32 | 画像ブロック表示   |

ヘッダー構造: `[type:1][seq:2][payload_len:4]` (ビッグエンディアン)

## 音声フォーマット
- サンプルレート: 16kHz
- ビット深度: 16bit signed PCM
- チャンネル: mono
- チャンクサイズ: 512サンプル (32ms)

## ハードウェアピン（M5StickS3）

| 機能        | GPIO |
|-------------|------|
| I2S MCLK    | 18   |
| I2S BCLK    | 17   |
| I2S WS      | 15   |
| I2S DOUT    | 14   |
| I2S DIN     | 16   |
| I2C SDA     | 47   |
| I2C SCL     | 48   |
| ES8311 I2C  | 0x18 |
| ボタンA     | 35   |
| LCD MOSI    | 39   |
| LCD SCLK    | 40   |
| LCD CS      | 41   |
| LCD DC      | 45   |
| LCD RST     | 21   |
| LCD BL      | 38   |

## LCD状態表示

| 色     | 状態       |
|--------|-----------|
| 黒     | IDLE       |
| 青     | LISTENING  |
| 黄     | PROCESSING |
| 緑     | SPEAKING   |

### ディスプレイ制御ツール
- `display_text`: 画面にテキスト表示（日本語対応、`size`で文字サイズ1-4を指定）
- `display_image`: 画像表示（`image_path` または `rgb565_base64`）
  - `image_path` 指定時はサーバー側でRGB565へ変換して表示
- RGB565は row-major / big-endian の生データ
- サーバー側で複数ブロックに分割して送信されるため、全画面画像も表示可能
- 日本語フォントを自動検出できない場合は `AICHATTER_DISPLAY_FONT` にフォントファイルパスを設定

## 動作確認

1. Ollama が起動していることを確認: `ollama list`
2. サーバーを起動: `cd server && uv run python main.py`
3. ファームウェアをフラッシュ: `idf.py -p /dev/cu.usbmodem1101 flash monitor`
4. 話しかける → LCDが青（LISTENING）→ 黄（PROCESSING）→ 緑（SPEAKING）
5. 応答中に割り込む → 即座に止まってLISTENING状態に遷移

## レイテンシ目標
- ASR: < 1秒
- LLM TTFT: < 2秒
- TTS: < 0.5秒
- エンドツーエンド: < 4秒
