# AiChatter - PC常時会話AI音声アシスタント

PCで動作するバージイン対応の常時会話型AI音声アシスタント。
すべてのAI処理はローカル実行（クラウドAPI不使用）。

## アーキテクチャ

```
PCサーバー (Python + uv)
┌────────────────────────────────┐
│ FastAPI WebSocket / REST API   │
│ AudioPipeline                  │
│  ├── Qwen3-ASR (ASR / mlx)    │
│  ├── Responses API互換LLM      │
│  ├── Qwen3-TTS (TTS / mlx)    │
│  ├── ツール (記憶/通知/検索等)  │
│  ├── 話者識別 (WavLM-XVector)  │
│  └── 自律行動エンジン           │
└────────────────────────────────┘
```

M5StickS3をクライアントとして接続することも可能です。→ [M5StickS3連携ガイド](docs/m5sticks3.md)

## 主な機能

- **音声対話**: ASR → LLM → TTS のパイプラインによるリアルタイム音声会話
- **バージイン**: 応答中でも割り込んで次の発話が可能
- **キャラクター設定**: YAMLで人格・声を定義。複数キャラクターの切替対応
- **ツール実行**: 記憶保存/検索、通知・リマインダー、Web検索、目標管理、画面表示、音量調整、スリープ制御
- **スキル動的注入**: ユーザー発言に応じてツール使用ガイドとメモリを自動でプロンプトに注入
- **自律行動**: キャラクターが自ら目標を管理し、目標に基づいて自律的に思考・行動・発話
- **サブエージェント**: 時間のかかる調査をバックグラウンドで実行し結果を通知
- **分散Web検索**: Tavily / Exa / Brave Searchの無料枠を優先的に活用する自動分散検索
- **グループモード**: 声紋で話者を識別し、複数人会話に対応（`--group`フラグで有効化）
- **REST API / CLI / SDK**: テキストベースでも利用可能

## セットアップ

### 必要なもの

- Python 3.10+ / uv
- OpenAI Responses API互換のLLMサーバー（LM Studio、Ollama等）

### サーバー準備

```bash
cd server
uv sync
cp .env.example .env
cp configs/character.example.yaml configs/character.yaml
cp configs/model.example.yaml configs/model.yaml
cp configs/prompt.example.yaml configs/prompt.yaml
```

#### 設定ファイル

| ファイル | 内容 |
|---------|------|
| `.env` | APIキー、サーバーポート、自律行動設定等 |
| `configs/character.yaml` | キャラクター（人格・声・シード目標）の定義 |
| `configs/model.yaml` | LLM・TTS・ASR・Embedding・Rerankの設定 |
| `configs/prompt.yaml` | 出力ルール・ツールガイド・スキル定義 |

#### キャラクター設定 (`configs/character.yaml`)

```yaml
persona:
  name: "アイ"
  system_prompt: |
    あなたは「アイ」という名前の明るく元気な女の子です。

  # シード目標（初回起動時にGoalStoreへ投入。以降はキャラクターが自分で管理）
  goals:
    - "ユーザーの興味に関連する情報を自分から調べて共有する"
    - "会話で気になったことを覚えておき、後で話題にする"

voice:
  # 方法1: テキスト説明から声を自動生成
  type: "description"
  description: "可愛らしい女性の声。高めのトーンで、明るく弾むような話し方。"

  # 方法2: WAVファイルで声を指定
  # type: "reference"
  # wav_file: "voices/my_voice.wav"
  # transcript: "参照音声に対応するテキスト"
```

#### モデル設定 (`configs/model.yaml`)

```yaml
# メインLLM（Responses API互換）
model: "qwen3.5-27b"
base_url: "http://localhost:1234/v1"
api_key: ""
reasoning: ""              # low, medium, high (空=未指定)

# サブエージェントLLM（省略時はメインと同じ）
sub:
  model: ""
  base_url: ""
  api_key: ""
  reasoning: ""

# Embedding API（メモリ検索の精度向上、省略可）
embeddings:
  model: ""
  base_url: ""
  api_key: ""
  dimensions: 0

# Rerank API（検索結果の再順位付け、省略可）
rerank:
  model: ""
  base_url: ""
  api_key: ""

# TTS設定（省略時はプラットフォームに応じたデフォルト値を使用）
tts:
  model: ""
  voice_design_model: ""

# ASR設定（省略時はプラットフォームに応じたデフォルト値を使用）
asr:
  model: ""
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

### CLIモード（音声対話）

PCのマイク/スピーカーを使って音声で対話します。

```bash
cd server

# 音声対話モード
uv run ai-chatter voice

# キャラクター指定
uv run ai-chatter voice -c character_custom.yaml

# グループモード（話者識別有効）
uv run ai-chatter voice --group
```

### CLIモード（テキスト対話）

```bash
cd server

# テキスト対話モード（ストリーミング表示）
uv run ai-chatter chat --stream

# TTS音声で返答を再生しながらテキスト対話
uv run ai-chatter chat --stream --voice
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

## 自律行動

キャラクターが自ら目標を設定・管理し、目標に基づいて自律的に思考・行動する機能です。

### 仕組み

1. `character.yaml` の `goals` でシード目標を定義（初回起動時のみ投入）
2. 以降はキャラクターが `add_goal` / `update_goal` / `complete_goal` ツールで自分の目標を管理
3. 自律行動スケジューラーが定期的にアクティブ目標をLLMに渡し、次の行動を判断させる
4. キャラクターはツール実行（調査、メモ保存等）やユーザーへの話しかけを自分で決定
5. 行動する必要がなければ `[PASS]` を出力してスキップ

### 設定 (`.env`)

```bash
AUTONOMOUS_ENABLED=true     # 自律行動の有効化
AUTONOMOUS_INTERVAL=600     # 思考間隔（秒、デフォルト: 600）
AUTONOMOUS_MAX_CONSECUTIVE=3  # ユーザー応答なしの連続行動上限
```

### 目標の構造

目標は `data/{キャラクター名}/goals.json` に永続化されます。

- **短期目標** (`short_term`): 「さっきの話題を調べる」など
- **長期目標** (`long_term`): 「ユーザーの趣味を深く理解する」など
- **ステータス**: `active` / `completed` / `paused`

## 分散Web検索

複数の検索エンジンAPIに対応し、無料枠を優先的に活用する自動分散検索を行います。

| エンジン | 無料枠 | 環境変数 |
|---------|--------|----------|
| Tavily | 1,000回/月 | `TAVILY_API_KEY` |
| Exa | 1,000回/月 | `EXA_API_KEY` |
| Brave Search | 2,000回/月 | `BRAVE_SEARCH_API_KEY` |

- APIキーが設定されたエンジンを自動検出
- 各エンジンの無料枠残量を追跡し、残り割合が高いエンジンから優先的に使用
- 失敗時は次のエンジンへ自動フォールバック
- LLMからは `web_search` ツール1つのみ見え、エンジン選択は内部で自動処理
- 使用量は `data/.cache/search_usage.json` に記録（日/月で自動リセット）

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
| `web_search` | Web検索（Tavily/Exa/Brave自動選択） | いずれかのAPIキー設定時 |
| `add_goal` | 目標を追加 | 常時 |
| `update_goal` | 目標を更新（進捗・ステータス） | 常時 |
| `list_goals` | 目標一覧を取得 | 常時 |
| `complete_goal` | 目標を完了にする | 常時 |
| `delete_goal` | 目標を削除 | 常時 |
| `set_notification` | 通知・リマインダー設定 | 常時 |
| `list_notifications` | 通知一覧 | 常時 |
| `delete_notification` | 通知削除 | 常時 |
| `get_battery` | バッテリー状態取得 | 常時 |
| `set_volume` | 音量調整 | 常時 |
| `set_sleep` | デバイスをスリープ | M5接続時 |
| `display_text` | 画面にテキスト表示 | M5接続時 |
| `display_image` | 画面に画像表示 | M5接続時 |
| `run_subagent_research` | バックグラウンド調査 | サブエージェント有効時 |
| `list_subagent_jobs` | 調査ジョブ一覧 | サブエージェント有効時 |
| `get_subagent_job` | 調査結果取得 | サブエージェント有効時 |
| `register_speaker` | 話者の声を登録 | グループモード時 |
| `list_speakers` | 登録済み話者一覧 | グループモード時 |
| `unregister_speaker` | 話者登録の解除 | グループモード時 |
| `merge_speakers` | 話者の統合 | グループモード時 |
