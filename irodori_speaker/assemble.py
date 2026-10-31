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
    path: Path | None  # kind == "break" は音声なし
    chapter_idx: int
    para_idx: int
    seg_idx: int
    kind: str = "narration"
    text: str = ""



def _silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def safe_name(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")[:40] or "chapter"


def _tmp_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.tmp.wav")


def times_path(chapter_wav: Path) -> Path:
    return chapter_wav.with_suffix(".times.json")


_ELLIPSIS_CHARS = "…‥―—"
_CLOSERS = "」』）)】"


def ends_with_ellipsis(text: str) -> bool:
    """文末が三点リーダ・ダッシュで終わるか（閉じ括弧の前でも）。"""
    return text.rstrip(_CLOSERS)[-1:] in tuple(_ELLIPSIS_CHARS)


def gap_seconds(prev: Clip, clip: Clip, silence: SilenceConfig, after_break: bool) -> float:
    """prev（音声のある直前のクリップ）から clip までの無音。基本と特別の大きい方。"""
    gap = silence.paragraph if clip.para_idx != prev.para_idx else silence.sentence
    if after_break:
        gap = max(gap, silence.scene)
    if prev.kind == "title":
        gap = max(gap, silence.title)
    if clip.para_idx == prev.para_idx and {prev.kind, clip.kind} == {"dialogue", "narration"}:
        gap = max(gap, silence.turn)
    if ends_with_ellipsis(prev.text):
        gap = max(gap, silence.ellipsis)
    return gap


def write_chapter(out_path: Path, clips: list[Clip], silence: SilenceConfig) -> int:
    """1 章分のクリップに無音を挟んで書き出し、サンプリングレートを返す。

    一時ファイルに書いてから置き換えるので、失敗しても既存の出力は壊れない。
    段落ごとの開始・終了サンプル（終了は最後の文の終わり）を `.times.json` にも書く。
    """
    audio = [c for c in clips if c.path is not None]
    if not audio:
        raise ValueError("音声のある文がありません")
    sr = sf.info(audio[0].path).samplerate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(out_path)
    times = times_path(out_path)
    frames = 0
    paragraphs: list[list[int]] = []
    segments: list[list[int]] = []
    try:
        with sf.SoundFile(tmp, "w", samplerate=sr, channels=1, subtype="PCM_16") as out:
            lead = _silence(silence.chapter, sr)
            out.write(lead)
            frames += len(lead)
            prev: Clip | None = None
            after_break = False
            for clip in clips:
                if clip.path is None:  # 区切りは幅 0 の段落として記録し、次の文の前の無音を長くする
                    paragraphs.append([clip.para_idx, frames, frames])
                    after_break = True
                    continue
                if prev is not None:
                    gap = _silence(gap_seconds(prev, clip, silence, after_break), sr)
                    out.write(gap)
                    frames += len(gap)
                after_break = False
                data, clip_sr = sf.read(clip.path, dtype="float32")
                if clip_sr != sr:
                    raise ValueError(f"サンプリングレートが一致しません: {clip.path}")
                if prev is None or clip.para_idx != prev.para_idx:
                    paragraphs.append([clip.para_idx, frames, frames])
                start = frames
                out.write(data)
                frames += len(data)
                paragraphs[-1][2] = frames
                segments.append([clip.para_idx, clip.seg_idx, start, frames])
                prev = clip
        tmp.replace(out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        times.unlink(missing_ok=True)
        raise
    tmp_times = times.with_name(times.name + ".tmp")
    try:
        tmp_times.write_text(
            json.dumps({"sr": sr, "frames": frames, "paragraphs": paragraphs, "segments": segments}),
            encoding="utf-8",
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
