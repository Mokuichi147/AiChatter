# AiChatter - M5StickS3 常時会話AI音声アシスタント

M5StickS3をバージイン対応の常時会話型AI音声アシスタントにするプロジェクト。
すべてのAI処理はローカル実行（クラウドAPI不使用）。

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

## 主な機能

- **音声対話**: ASR → LLM → TTS のパイプラインによるリアルタイム音声会話
- **バージイン**: 応答中でも割り込んで次の発話が可能
- **キャラクター設定**: YAMLで人格・声を定義。複数キャラクターの切替対応
- **ツール実行**: 記憶保存/検索、通知・リマインダー、Web検索、画面表示、音量調整、スリープ制御
- **スキル動的注入**: ユーザー発言に応じてツール使用ガイドとメモリを自動でプロンプトに注入
- **サブエージェント**: 時間のかかる調査をバックグラウンドで実行し結果を通知
- **グループモード**: 声紋で話者を識別し、複数人会話に対応（`--group`フラグで有効化）
- **REST API / CLI / SDK**: M5StickS3未接続でもテキストベースで利用可能

## セットアップ

### 必要なもの
- M5StickS3 (ESP32-S3)
- Python 3.10+ / uv
- ESP-IDF 5.3
- OpenAI Responses API互換のLLMサーバー（Ollama等）

### サーバー準備

```bash
cd server
uv sync
cp .env.example .env
cp configs/character.example.yaml configs/character.yaml
cp configs/llm.example.yaml configs/llm.yaml
cp configs/prompt.example.yaml configs/prompt.yaml
```

#### 設定ファイル

| ファイル | 内容 |
|---------|------|
| `.env` | ASRモデル、APIキー、サーバーポート等 |
| `configs/character.yaml` | キャラクター（人格・声）の定義 |
| `configs/llm.yaml` | LLM・Embedding・Rerankの接続設定 |
| `configs/prompt.yaml` | 出力ルール・ツールガイド・スキル定義 |

#### キャラクター設定 (`configs/character.yaml`)

```yaml
persona:
  name: "アイ"
  system_prompt: |
    あなたは「アイ」という名前の明るく元気な女の子です。

voice:
  # 方法1: テキスト説明から声を自動生成
  type: "description"
  description: "可愛らしい女性の声。高めのトーンで、明るく弾むような話し方。"

  # 方法2: WAVファイルで声を指定
  # type: "reference"
  # wav_file: "voices/my_voice.wav"
  # transcript: "参照音声に対応するテキスト"
```

#### LLM設定 (`configs/llm.yaml`)

```yaml
# メインLLM（Responses API互換）
model: "qwen2.5:7b"
base_url: "http://localhost:11434/v1"
api_key: ""

# サブエージェントLLM（省略時はメインと同じ）
sub:
  model: ""
  base_url: ""

# Embedding API（メモリ検索の精度向上、省略可）
embeddings:
  model: "text-embedding-3-small"
  base_url: ""
  dimensions: 0

# Rerank API（検索結果の再順位付け、省略可）
rerank:
  model: "jina-reranker-v2-base-multilingual"
  base_url: ""
```

### サーバー起動

```bash
cd server
uv run ai-chatter server
```

#### オプション

```bash
# キャラクター指定
uv run ai-chatter server -c character_custom.yaml

# グループモード（話者識別有効）
uv run ai-chatter server --group
```

### CLIモード（テキスト対話）

```bash
cd server

# 対話モード
uv run ai-chatter chat --stream

# サーバー起動
uv run ai-chatter server -c 'character*.yaml'

# グループモードでサーバー起動
uv run ai-chatter server --group
```

### ファームウェアビルド

```bash
# ESP-IDF環境をロード
source ~/esp/esp-idf/export.sh

cd firmware

# WiFi/サーバーIP設定
nano main/config.h

idf.py set-target esp32s3
idf.py build
```

### REST API

```bash
# セッション作成
curl -X POST http://127.0.0.1:8765/api/v1/sessions \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","history_mode":"isolated"}'

# チャット（同期）
curl -X POST http://127.0.0.1:8765/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","text":"こんにちは"}'

# チャット（SSEストリーミング）
curl -N -X POST http://127.0.0.1:8765/api/v1/chat/stream \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","text":"今日の予定を教えて"}'

# キャラクター一覧
curl http://127.0.0.1:8765/api/v1/characters
```

### Python SDK

```python
import asyncio
from ai_chatter.aichatter import create_runtime

async def main():
    runtime = await create_runtime()
    await runtime.create_session("demo", history_mode="isolated")
    result = await runtime.chat("demo", "自己紹介して")
    print(result["text"])

asyncio.run(main())
```

## グループモード（複数人会話）

`--group`フラグで起動すると、声紋（WavLM-XVector）による話者識別が有効になります。

- 各発言に `[話者名]` プレフィックスが付与され、LLMが誰の発言か認識
- 未登録者は `[不明な人A]` `[不明な人B]` と自動でクラスタリング
- 名乗り（「私は太郎だよ」）を検出すると自動で声を登録
- 他者同士の会話には `[SKIP]` を返しTTS合成をスキップ
- `.env`の`SPEAKER_SIMILARITY_THRESHOLD`で識別閾値を調整可能（デフォルト: 0.65）

### 話者管理ツール

LLMが自動で使用するツールとして、以下が利用可能です:

- `register_speaker`: 今話している人の声を名前で登録
- `list_speakers`: 登録済み話者の一覧
- `unregister_speaker`: 話者登録の解除
- `merge_speakers`: 同一人物が別名で登録された場合の統合

## ツール一覧

| ツール | 機能 | 利用条件 |
|--------|------|----------|
| `save_memory` | 情報を記憶に保存 | 常時 |
| `search_memory` | 記憶を検索 | 常時 |
| `delete_memory` | 記憶を削除 | 常時 |
| `search` | Web検索 | `TAVILY_API_KEY`設定時 |
| `set_notification` | 通知・リマインダー設定 | 常時 |
| `list_notifications` | 通知一覧 | 常時 |
| `delete_notification` | 通知削除 | 常時 |
| `set_volume` | 音量調整 | 常時 |
| `set_sleep` | デバイスをスリープ | M5接続時 |
| `display_text` | 画面にテキスト表示 | M5接続時 |
| `display_image` | 画面に画像表示 | M5接続時 |
| `run_subagent_research` | バックグラウンド調査 | サブエージェント有効時 |
| `register_speaker` | 話者の声を登録 | グループモード時 |

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
