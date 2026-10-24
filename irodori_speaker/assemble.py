from __future__ import annotations

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


def write_chapter(out_path: Path, clips: list[Clip], silence: SilenceConfig) -> int:
    """1 章分のクリップに無音を挟んで書き出し、サンプリングレートを返す。"""
    sr = sf.info(clips[0].path).samplerate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(out_path, "w", samplerate=sr, channels=1, subtype="PCM_16") as out:
        out.write(_silence(silence.chapter, sr))
        prev: Clip | None = None
        for clip in clips:
            if prev is not None:
                gap = silence.paragraph if clip.para_idx != prev.para_idx else silence.sentence
                out.write(_silence(gap, sr))
            data, clip_sr = sf.read(clip.path, dtype="float32")
            if clip_sr != sr:
                raise ValueError(f"サンプリングレートが一致しません: {clip.path}")
            out.write(data)
            prev = clip
    return sr


def concat_wavs(paths: list[Path], out_path: Path) -> None:
    if not paths:
        return
    sr = sf.info(paths[0]).samplerate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(out_path, "w", samplerate=sr, channels=1, subtype="PCM_16") as out:
        for p in paths:
            for block in sf.blocks(p, blocksize=sr * 30, dtype="float32"):
                out.write(block)
