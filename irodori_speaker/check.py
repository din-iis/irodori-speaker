"""合成済みの文の品質チェック（長さ・前後の無音・末尾の孤立音・クリッピング、任意で ASR の文字誤り率）。

numpy / soundfile だけで動く。transformers は transcribe の中でだけ import する。
"""

from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .assemble import times_path
from .emoji import emoji_runs
from .synth import cache_key, row_seed, tts_text

CHECK_FIELDS = [
    "chapter", "para", "seg", "key", "seconds", "chars", "sec_per_char", "z", "flags", "offset", "text",
]
ASR_FIELDS = ["cer", "transcript"]

# 実データで調整が要るので設定には入れない
FRAME_SECONDS = 0.02
SILENCE_DB = -40.0
TAIL_MAX_SECONDS = 0.35
TAIL_GAP_SECONDS = 0.4
CLIP_LEVEL = 0.99


@dataclass
class AudioStats:
    seconds: float
    leading_silence: float
    trailing_silence: float
    tail_noise: float
    clipping: int


@dataclass
class CheckRow:
    chapter: int
    para: int
    seg: int
    key: str
    text: str
    chars: int
    stats: AudioStats | None
    z: float | None
    offset: float | None
    flags: list[str] = field(default_factory=list)
    cer: float | None = None
    transcript: str | None = None


# ---- 文字数 ----


def strip_emoji(text: str) -> str:
    out, pos = [], 0
    for start, end in emoji_runs(text):
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def count_chars(text: str) -> int:
    """モデルが実際に読む文字の数（絵文字・文末の句点・空白を除く）。"""
    return len("".join(strip_emoji(tts_text(text)).split()))


# ---- 音声の解析 ----


def frame_rms_db(data: np.ndarray, sr: int) -> np.ndarray:
    """FRAME_SECONDS ごとの RMS（dBFS）。端数のフレームは捨てる。"""
    size = max(1, int(round(FRAME_SECONDS * sr)))
    n = len(data) // size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = data[: n * size].astype(np.float64).reshape(n, size)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def speech_runs(db: np.ndarray) -> list[tuple[int, int]]:
    """SILENCE_DB を超えるフレームの連なり [start, end)。"""
    runs: list[tuple[int, int]] = []
    start = None
    for i, loud in enumerate(db > SILENCE_DB):
        if loud and start is None:
            start = i
        elif not loud and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(db)))
    return runs


def analyze_audio(data: np.ndarray, sr: int) -> AudioStats:
    seconds = len(data) / sr
    runs = speech_runs(frame_rms_db(data, sr))
    if not runs:
        return AudioStats(seconds, seconds, seconds, 0.0, 0)
    leading = runs[0][0] * FRAME_SECONDS
    trailing = max(0.0, seconds - runs[-1][1] * FRAME_SECONDS)
    tail = 0.0
    if len(runs) >= 2:
        last_len = (runs[-1][1] - runs[-1][0]) * FRAME_SECONDS
        gap = (runs[-1][0] - runs[-2][1]) * FRAME_SECONDS
        if last_len <= TAIL_MAX_SECONDS and gap >= TAIL_GAP_SECONDS - 1e-9:
            tail = last_len
    clipping = int(np.count_nonzero(np.abs(data) >= CLIP_LEVEL))
    return AudioStats(seconds, leading, trailing, tail, clipping)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def analyze_file(path: Path) -> AudioStats | None:
    path = Path(path)
    if not path.exists():
        return None
    data, sr = _read_mono(path)
    return analyze_audio(data, sr)


def robust_z(values: list[float | None]) -> list[float | None]:
    """中央値と MAD×1.4826 によるロバストな z。MAD が 0 なら 0、None は飛ばす。"""
    present = np.array([v for v in values if v is not None], dtype=np.float64)
    if present.size == 0:
        return [None for _ in values]
    median = float(np.median(present))
    mad = float(np.median(np.abs(present - median))) * 1.4826
    return [None if v is None else (0.0 if mad == 0 else (v - median) / mad) for v in values]


# ---- 印 ----


def flags_for(row: CheckRow, z_threshold: float, cer_threshold: float, warned: set[str]) -> list[str]:
    flags: list[str] = []
    st = row.stats
    if row.z is not None and row.z >= z_threshold:
        flags.append("long")
    if row.z is not None and row.z <= -z_threshold:
        flags.append("short")
    if st is not None and st.clipping > 0:
        flags.append("clip")
    if st is not None and st.tail_noise > 0:
        flags.append("tail")
    if st is None:
        flags.append("missing")
    if row.key in warned:
        flags.append("warned")
    if row.cer is not None and row.cer >= cer_threshold:
        flags.append("cer")
    return flags


# ---- キー・警告・時刻 ----


def segment_key(synth, row: dict) -> str:
    """その行の文が segments/ に保存されるときのキー。synthesize と同じ計算に寄せる。"""
    return cache_key(synth.cfg, row["text"], synth.latent_key, row_seed(row))


def read_warned_keys(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(line.split("\t", 1)[0])
    return keys


def load_times_json(chapter_wav: Path) -> dict | None:
    path = times_path(Path(chapter_wav))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def offset_for(times: dict | None, para: int, seg: int) -> float | None:
    """章 WAV の中でその文（無ければ段落）が始まる秒。"""
    if not times or not times.get("sr"):
        return None
    sr = times["sr"]
    for p, s, start, _end in times.get("segments", []):
        if (p, s) == (para, seg):
            return start / sr
    for p, start, _end in times.get("paragraphs", []):
        if p == para:
            return start / sr
    return None


# ---- ASR ----


def normalize_for_cer(text: str) -> str:
    """NFKC にして、絵文字・句読点・記号・空白を除く。"""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        if unicodedata.category(ch)[0] in "PSZC":
            continue
        out.append(ch)
    return "".join(out)


def cer(ref: str, hyp: str) -> float:
    """文字誤り率（編集距離 / 参照の長さ）。参照が空なら 0。"""
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def transcribe(paths: list[Path], model: str, device: str, batch_size: int = 8) -> list[str]:
    """音声を文字に起こす。transformers は ここでだけ import する（Mac には無い）。"""
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model, device=device)
    try:
        inputs = []
        for p in paths:
            data, sr = _read_mono(Path(p))
            inputs.append({"raw": data, "sampling_rate": sr})
        results = asr(
            inputs, batch_size=batch_size, generate_kwargs={"language": "ja", "task": "transcribe"}
        )
        return [r["text"] for r in results]
    finally:
        del asr
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def transcribe_with(check_cfg) -> Callable[[list[Path]], list[str]]:
    """CheckConfig の ASR 設定で paths を文字に起こす関数を返す。"""

    def run(paths: list[Path]) -> list[str]:
        return transcribe(paths, check_cfg.asr_model, check_cfg.asr_device)

    return run


# ---- check.tsv ----


def _fmt(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)


def write_check_tsv(path: Path, rows: list[CheckRow], asr: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = CHECK_FIELDS + (ASR_FIELDS if asr else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            seconds = r.stats.seconds if r.stats else None
            per_char = seconds / r.chars if seconds is not None and r.chars else None
            row = {
                "chapter": r.chapter, "para": r.para, "seg": r.seg, "key": r.key,
                "seconds": _fmt(seconds, ".2f"), "chars": r.chars,
                "sec_per_char": _fmt(per_char, ".3f"), "z": _fmt(r.z, ".2f"),
                "flags": "|".join(r.flags), "offset": _fmt(r.offset, ".2f"), "text": r.text,
            }
            if asr:
                row["cer"] = _fmt(r.cer, ".3f")
                row["transcript"] = r.transcript or ""
            w.writerow(row)
