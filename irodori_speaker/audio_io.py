from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_FFMPEG_MISSING = "ffmpeg が見つかりません。インストールしてください"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(args: list[str]) -> bytes:
    if not ffmpeg_available():
        raise RuntimeError(_FFMPEG_MISSING)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", *args],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg が失敗しました: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout


def decode(
    path: Path,
    sr: int,
    channels: int,
    start: float | None = None,
    seconds: float | None = None,
) -> np.ndarray:
    """音声を float32 で読む。channels=1 は (n,)、2 は (n, 2)。"""
    args: list[str] = []
    if start is not None:
        args += ["-ss", str(start)]
    args += ["-i", str(path)]
    if seconds is not None:
        args += ["-t", str(seconds)]
    args += ["-ar", str(sr), "-ac", str(channels), "-f", "f32le", "-"]
    audio = np.frombuffer(_run(args), dtype="<f4").astype(np.float32)
    return audio if channels == 1 else audio.reshape(-1, channels)


def write_mp3(audio: np.ndarray, sr: int, path: Path, bitrate: str) -> None:
    """一時ファイル経由で MP3 を書く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.mp3")
    try:
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "in.wav"
            sf.write(wav, audio, sr, subtype="FLOAT")
            _run(["-y", "-i", str(wav), "-b:a", bitrate, "-f", "mp3", str(tmp)])
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _concat_list(paths: list[Path]) -> str:
    return "".join("file '{}'\n".format(str(p.resolve()).replace("'", "'\\''")) for p in paths)


def concat_mp3(paths: list[Path], out: Path) -> None:
    """MP3 を再エンコードせずに連結する。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.tmp.mp3")
    try:
        with tempfile.TemporaryDirectory() as d:
            lst = Path(d) / "list.txt"
            lst.write_text(_concat_list(paths), encoding="utf-8")
            _run(["-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-f", "mp3", str(tmp)])
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def encode_file(src: Path, dst: Path, args: list[str]) -> None:
    """src を ffmpeg の args でエンコードして dst に書く（一時ファイル経由）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        _run(["-y", "-i", str(src), *args, str(tmp)])
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_m4b(inputs: list[Path], metadata: str, out: Path, bitrate: str, channels: int) -> None:
    """章 WAV / MP3 を連結して AAC にし、ffmetadata の章マーカーを付けた M4B を書く。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.tmp.m4b")
    try:
        with tempfile.TemporaryDirectory() as d:
            lst, meta = Path(d) / "list.txt", Path(d) / "meta.txt"
            lst.write_text(_concat_list(inputs), encoding="utf-8")
            meta.write_text(metadata, encoding="utf-8")
            _run([
                "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-f", "ffmetadata", "-i", str(meta),
                "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1",
                "-c:a", "aac", "-b:a", bitrate, "-ac", str(channels),
                "-movflags", "+faststart", "-f", "ipod", str(tmp),
            ])
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
