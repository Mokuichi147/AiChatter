"""話者識別精度の評価スクリプト。

TTS (VoiceDesign + Base) で複数話者の音声サンプルを合成し、
SpeakerIdentifier の識別精度を定量評価する。

使用方法:
    cd server
    uv run python tests/eval_speaker_id.py [--generate] [--threshold 0.65]

    --generate  : テスト音声を再生成する（キャッシュがあればスキップ）
    --threshold : 類似度閾値（デフォルト: 0.65）
    --enroll N  : 登録用サンプル数（デフォルト: 2）
    --seed N    : 乱数シード（デフォルト: 42）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# プロジェクトルート
SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT / "src"))

CACHE_DIR = SERVER_ROOT / "tests" / ".cache" / "speaker_eval"
SAMPLE_RATE = 16000  # SpeakerIdentifier の入力サンプルレート
DEFAULT_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 話者プロファイル定義 ──────────────────────────────────────

@dataclass
class SpeakerProfile:
    """テスト話者の定義。"""
    name: str
    description: str  # VoiceDesign の instruct
    sample_text: str  # VoiceDesign 用テキスト


# 声質が異なる6種類の話者を定義
SPEAKERS = [
    SpeakerProfile(
        name="若い女性A",
        description="若い女性の声。高めのピッチで明るく弾むような話し方。",
        sample_text="こんにちは、今日はとてもいい天気ですね。",
    ),
    SpeakerProfile(
        name="若い女性B",
        description="若い女性の声。落ち着いたトーンで丁寧に話す。",
        sample_text="お疲れ様です、本日の会議の資料を準備しました。",
    ),
    SpeakerProfile(
        name="成人男性A",
        description="成人男性の低い声。ゆっくりと落ち着いた話し方。",
        sample_text="なるほど、それは面白い考えですね。",
    ),
    SpeakerProfile(
        name="成人男性B",
        description="成人男性の声。やや高めで早口な話し方。",
        sample_text="了解しました、すぐに対応します。",
    ),
    SpeakerProfile(
        name="年配女性",
        description="年配の女性の声。穏やかで優しい口調。",
        sample_text="まあ、それは大変でしたね。ゆっくり休んでくださいね。",
    ),
    SpeakerProfile(
        name="年配男性",
        description="年配の男性の声。渋くて威厳のあるトーン。",
        sample_text="昔はこういうことがよくあったものだ。",
    ),
]

# 各話者の評価用発話テキスト
UTTERANCE_TEXTS = [
    "明日の天気はどうなるかな。",
    "最近読んだ本がとても面白かった。",
    "お腹がすいたので何か食べに行きましょう。",
    "この問題の解決策を考えてみましょう。",
    "週末に映画を見に行く予定です。",
]


# ── 音声生成 ─────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """再現性のため全乱数生成器のシードを固定する。"""
    import random
    import mlx.core as mx

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.info(f"乱数シード設定: {seed}")


def _cache_key(description: str, text: str) -> str:
    return hashlib.md5(f"{description}:{text}".encode()).hexdigest()[:12]


def generate_reference_voice(
    design_model, description: str, sample_text: str
) -> tuple[str, int]:
    """VoiceDesign で参照音声を生成し、WAVファイルパスを返す。"""
    import soundfile as sf

    key = _cache_key(description, sample_text)
    cache_path = CACHE_DIR / f"ref_{key}.wav"

    if cache_path.exists():
        info = sf.info(str(cache_path))
        return str(cache_path), info.samplerate

    logger.info(f"  参照音声生成中: {description[:40]}...")
    segments = []
    for result in design_model.generate(
        text=sample_text,
        instruct=description,
        lang_code="Japanese",
        temperature=0.0,
    ):
        if result.audio is not None:
            segments.append(np.array(result.audio, dtype=np.float32))

    if not segments:
        raise RuntimeError(f"参照音声の生成に失敗: {description}")

    audio = np.concatenate(segments)
    audio = np.clip(audio, -1.0, 1.0)
    sr = getattr(design_model, "sample_rate", 24000)

    sf.write(str(cache_path), audio, sr)
    return str(cache_path), sr


def generate_utterance(
    tts_model, ref_audio: str, ref_text: str, text: str, model_sr: int
) -> bytes:
    """Base TTS で発話を合成し、16kHz 16bit PCM bytes を返す。"""
    from scipy.signal import resample_poly

    key = _cache_key(f"{ref_audio}:{ref_text}", text)
    cache_path = CACHE_DIR / f"utt_{key}.pcm"

    if cache_path.exists():
        return cache_path.read_bytes()

    segments = []
    for result in tts_model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        lang_code="Japanese",
        temperature=0.0,
    ):
        if result.audio is not None:
            segments.append(np.array(result.audio, dtype=np.float32))

    if not segments:
        raise RuntimeError(f"発話合成失敗: {text}")

    audio = np.concatenate(segments)

    # リサンプル: model_sr → 16kHz
    if model_sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, model_sr).astype(np.float32)

    # 正規化 → 16bit PCM
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    pcm = (audio * 32767).astype(np.int16).tobytes()

    cache_path.write_bytes(pcm)
    return pcm


def generate_all_samples(speakers: list[SpeakerProfile]) -> dict[str, list[bytes]]:
    """全話者の全発話サンプルを生成する。キャッシュ済みならスキップ。"""
    from ai_chatter.config import tts_config

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # VoiceDesign モデル
    design_model_name = tts_config.get_voice_design_model()
    logger.info(f"VoiceDesign モデル読み込み中: {design_model_name}")
    from mlx_audio.tts.utils import load_model
    design_model = load_model(design_model_name)

    # Base TTS モデル
    tts_model_name = tts_config.get_model()
    logger.info(f"Base TTS モデル読み込み中: {tts_model_name}")
    tts_model = load_model(tts_model_name)
    tts_sr = getattr(tts_model, "sample_rate", 24000)

    samples: dict[str, list[bytes]] = {}

    for speaker in speakers:
        logger.info(f"話者 [{speaker.name}] のサンプル生成中...")

        # 参照音声生成
        ref_wav, ref_sr = generate_reference_voice(
            design_model, speaker.description, speaker.sample_text
        )

        # 各テキストで発話合成
        utterances: list[bytes] = []
        for i, text in enumerate(UTTERANCE_TEXTS):
            logger.info(f"  発話 {i+1}/{len(UTTERANCE_TEXTS)}: {text[:20]}...")
            pcm = generate_utterance(tts_model, ref_wav, speaker.sample_text, text, tts_sr)
            utterances.append(pcm)

        samples[speaker.name] = utterances
        logger.info(f"  → {len(utterances)} サンプル生成完了")

    # VoiceDesign モデルを解放
    del design_model

    return samples


# ── 評価 ─────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """評価結果。"""
    num_speakers: int = 0
    num_enroll: int = 0
    num_test: int = 0
    threshold: float = 0.65
    # 識別結果
    correct: int = 0
    wrong: int = 0
    rejected: int = 0  # 登録済みなのに「不明」と判定
    # スコア分布
    same_speaker_scores: list[float] = field(default_factory=list)
    diff_speaker_scores: list[float] = field(default_factory=list)
    # 混同行列: confusion[true_speaker][predicted] = count
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    # 各話者のスコア詳細
    detail_log: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = self.correct + self.wrong + self.rejected
        return self.correct / total if total > 0 else 0.0

    @property
    def rejection_rate(self) -> float:
        total = self.correct + self.wrong + self.rejected
        return self.rejected / total if total > 0 else 0.0

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("話者識別 評価レポート")
        print("=" * 70)
        print(f"話者数:         {self.num_speakers}")
        print(f"登録サンプル数:  {self.num_enroll} / 話者")
        print(f"テストサンプル数: {self.num_test}")
        print(f"類似度閾値:      {self.threshold}")
        print()
        print(f"正答:   {self.correct}/{self.num_test} ({self.correct/max(self.num_test,1)*100:.1f}%)")
        print(f"誤答:   {self.wrong}/{self.num_test} ({self.wrong/max(self.num_test,1)*100:.1f}%)")
        print(f"棄却:   {self.rejected}/{self.num_test} ({self.rejected/max(self.num_test,1)*100:.1f}%)")
        print(f"精度:   {self.accuracy*100:.1f}%")
        print()

        # スコア分布
        if self.same_speaker_scores:
            same = np.array(self.same_speaker_scores)
            print(f"同一話者スコア: mean={same.mean():.4f}, std={same.std():.4f}, "
                  f"min={same.min():.4f}, max={same.max():.4f}")
        if self.diff_speaker_scores:
            diff = np.array(self.diff_speaker_scores)
            print(f"異話者スコア:   mean={diff.mean():.4f}, std={diff.std():.4f}, "
                  f"min={diff.min():.4f}, max={diff.max():.4f}")

        # EER 推定
        if self.same_speaker_scores and self.diff_speaker_scores:
            eer, eer_threshold = self._compute_eer()
            print(f"\nEER (Equal Error Rate): {eer*100:.2f}% (閾値={eer_threshold:.4f})")

        # 混同行列
        print("\n--- 混同行列 ---")
        speakers = sorted(self.confusion.keys())
        # ヘッダー
        max_name_len = max(len(s) for s in speakers) if speakers else 0
        header = " " * (max_name_len + 2) + "  ".join(f"{s[:6]:>6}" for s in speakers) + "  棄却"
        print(header)
        for true_sp in speakers:
            row = f"{true_sp:<{max_name_len}}  "
            for pred_sp in speakers:
                count = self.confusion.get(true_sp, {}).get(pred_sp, 0)
                row += f"{count:>6}  "
            # 棄却数
            rej = self.confusion.get(true_sp, {}).get("__rejected__", 0)
            row += f"{rej:>4}"
            print(row)

        # 詳細ログ
        if self.detail_log:
            print("\n--- 詳細スコアログ ---")
            for line in self.detail_log:
                print(line)

    def _compute_eer(self) -> tuple[float, float]:
        """同一話者/異話者スコアからEERと最適閾値を推定する。"""
        same = np.array(self.same_speaker_scores)
        diff = np.array(self.diff_speaker_scores)

        thresholds = np.linspace(0.0, 1.0, 1000)
        best_eer = 1.0
        best_thr = 0.5

        for thr in thresholds:
            # FRR: 同一話者なのに閾値未満（棄却）
            frr = np.mean(same < thr)
            # FAR: 異話者なのに閾値以上（誤受理）
            far = np.mean(diff >= thr)

            if abs(frr - far) < abs(best_eer - 0.5):
                eer_candidate = (frr + far) / 2
                if abs(frr - far) < abs(best_eer * 2 - frr - far):
                    best_eer = eer_candidate
                    best_thr = thr

        # より正確なEER計算
        min_diff = float("inf")
        for thr in thresholds:
            frr = np.mean(same < thr)
            far = np.mean(diff >= thr)
            d = abs(frr - far)
            if d < min_diff:
                min_diff = d
                best_eer = (frr + far) / 2
                best_thr = float(thr)

        return best_eer, best_thr


def evaluate(
    samples: dict[str, list[bytes]],
    num_enroll: int,
    threshold: float,
) -> EvalResult:
    """話者識別の精度を評価する。"""
    import tempfile
    from ai_chatter.speaker_id import SpeakerIdentifier

    result = EvalResult(
        num_speakers=len(samples),
        num_enroll=num_enroll,
        threshold=threshold,
    )

    # 一時ファイルで SpeakerIdentifier を初期化（既存データを汚さない）
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "speakers.json"
        sid = SpeakerIdentifier(str(data_path), similarity_threshold=threshold)

        # 登録フェーズ
        logger.info(f"登録フェーズ: 各話者 {num_enroll} サンプルを登録")
        for speaker_name, utterances in samples.items():
            for i in range(min(num_enroll, len(utterances))):
                sid.enroll(speaker_name, utterances[i])
            logger.info(f"  {speaker_name}: {min(num_enroll, len(utterances))} サンプル登録")

        # テストフェーズ
        speaker_names = list(samples.keys())
        test_count = 0

        for true_speaker, utterances in samples.items():
            test_utterances = utterances[num_enroll:]
            if not test_utterances:
                logger.warning(f"  {true_speaker}: テスト用サンプルなし（全て登録に使用）")
                continue

            result.confusion.setdefault(true_speaker, {})

            for i, pcm in enumerate(test_utterances):
                test_count += 1

                # embedding 計算
                embedding = sid.compute_embedding(pcm)

                # 全登録話者とのスコアを計算
                scores: dict[str, float] = {}
                for name in speaker_names:
                    centroid = np.mean(sid._speakers[name], axis=0)
                    score = sid._cosine_similarity(embedding, centroid)
                    scores[name] = score

                    if name == true_speaker:
                        result.same_speaker_scores.append(score)
                    else:
                        result.diff_speaker_scores.append(score)

                # 識別結果
                best_name = max(scores, key=scores.get)
                best_score = scores[best_name]

                if best_score < threshold:
                    # 棄却（閾値未満）
                    result.rejected += 1
                    result.confusion[true_speaker]["__rejected__"] = \
                        result.confusion[true_speaker].get("__rejected__", 0) + 1
                    verdict = "棄却"
                elif best_name == true_speaker:
                    result.correct += 1
                    result.confusion[true_speaker][best_name] = \
                        result.confusion[true_speaker].get(best_name, 0) + 1
                    verdict = "正答"
                else:
                    result.wrong += 1
                    result.confusion[true_speaker][best_name] = \
                        result.confusion[true_speaker].get(best_name, 0) + 1
                    verdict = "誤答"

                # スコア詳細
                scores_str = ", ".join(f"{n}={s:.4f}" for n, s in sorted(scores.items()))
                result.detail_log.append(
                    f"  [{verdict}] 正解={true_speaker}, 判定={best_name}({best_score:.4f}) "
                    f"| {scores_str}"
                )

        result.num_test = test_count

    return result


# ── メイン ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="話者識別精度の評価")
    parser.add_argument("--generate", action="store_true", help="テスト音声を強制再生成")
    parser.add_argument("--threshold", type=float, default=0.65, help="類似度閾値")
    parser.add_argument("--enroll", type=int, default=2, help="登録用サンプル数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="乱数シード")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.generate:
        # キャッシュクリア
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        logger.info("キャッシュをクリアしました")

    # 音声サンプル生成
    t0 = time.monotonic()
    logger.info(f"テスト音声サンプルの準備中（{len(SPEAKERS)}話者 × {len(UTTERANCE_TEXTS)}発話）...")
    samples = generate_all_samples(SPEAKERS)
    t1 = time.monotonic()
    logger.info(f"音声サンプル準備完了: {t1-t0:.1f}秒")

    # 評価実行
    logger.info(f"評価開始（閾値={args.threshold}, 登録数={args.enroll}）...")
    t2 = time.monotonic()
    result = evaluate(samples, num_enroll=args.enroll, threshold=args.threshold)
    t3 = time.monotonic()
    logger.info(f"評価完了: {t3-t2:.1f}秒")

    result.print_report()

    # 複数閾値での比較
    print("\n--- 閾値別精度 ---")
    print(f"{'閾値':>6}  {'正答率':>7}  {'棄却率':>7}  {'誤答率':>7}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        r = evaluate(samples, num_enroll=args.enroll, threshold=thr)
        total = max(r.num_test, 1)
        print(
            f"  {thr:.2f}  {r.correct/total*100:>6.1f}%  "
            f"{r.rejected/total*100:>6.1f}%  {r.wrong/total*100:>6.1f}%"
        )


if __name__ == "__main__":
    main()
