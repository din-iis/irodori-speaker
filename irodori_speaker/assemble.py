from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SilenceConfig


@dataclass
class Clip:
    path: Path
    chapter_idx: int
    para_idx: int
    seg_idx: int


def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def safe_name(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")[:40] or "chapter"


def _tmp_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.tmp.wav")


def times_path(chapter_wav: Path) -> Path:
    return chapter_wav.with_suffix(".times.json")


def write_chapter(out_path: Path, clips: list[Clip], silence: SilenceConfig) -> int:
    """1 章分のクリップに無音を挟んで書き出し、サンプリングレートを返す。

    一時ファイルに書いてから置き換えるので、失敗しても既存の出力は壊れない。
    段落ごとの開始・終了サンプル（終了は最後の文の終わり）を `.times.json` にも書く。
    """
    sr = sf.info(clips[0].path).samplerate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(out_path)
    times = times_path(out_path)
    frames = 0
    paragraphs: list[list[int]] = []
    try:
        with sf.SoundFile(tmp, "w", samplerate=sr, channels=1, subtype="PCM_16") as out:
            lead = _silence(silence.chapter, sr)
            out.write(lead)
            frames += len(lead)
            prev: Clip | None = None
            for clip in clips:
                if prev is not None:
                    gap = _silence(silence.paragraph if clip.para_idx != prev.para_idx else silence.sentence, sr)
                    out.write(gap)
                    frames += len(gap)
                data, clip_sr = sf.read(clip.path, dtype="float32")
                if clip_sr != sr:
                    raise ValueError(f"サンプリングレートが一致しません: {clip.path}")
                if prev is None or clip.para_idx != prev.para_idx:
                    paragraphs.append([clip.para_idx, frames, frames])
                out.write(data)
                frames += len(data)
                paragraphs[-1][2] = frames
                prev = clip
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        times.unlink(missing_ok=True)
        raise
    tmp_times = times.with_name(times.name + ".tmp")
    try:
        tmp_times.write_text(
            json.dumps({"sr": sr, "frames": frames, "paragraphs": paragraphs}), encoding="utf-8"
        )
        tmp_times.replace(times)
    except BaseException:
        tmp_times.unlink(missing_ok=True)
        times.unlink(missing_ok=True)
        raise
    return sr


def concat_wavs(paths: list[Path], out_path: Path) -> None:
    if not paths:
        return
    sr = sf.info(paths[0]).samplerate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(out_path)
    try:
        with sf.SoundFile(tmp, "w", samplerate=sr, channels=1, subtype="PCM_16") as out:
            for p in paths:
                if sf.info(p).samplerate != sr:
                    raise ValueError(f"サンプリングレートが一致しません: {p}")
                for block in sf.blocks(p, blocksize=sr * 30, dtype="float32"):
                    out.write(block)
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
