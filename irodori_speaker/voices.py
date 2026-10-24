from __future__ import annotations

import hashlib
from pathlib import Path

import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm"}


def _is_audio(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in AUDIO_EXTS


def _dir_clips(d: Path) -> list[Path]:
    return sorted(
        (p for p in d.iterdir() if not p.name.startswith(".") and _is_audio(p)),
        key=lambda p: p.name,
    )


def list_voices(voices_dir: Path) -> list[str]:
    voices_dir = Path(voices_dir)
    if not voices_dir.is_dir():
        return []
    ids: set[str] = set()
    for p in voices_dir.iterdir():
        if p.name.startswith("."):
            continue
        if _is_audio(p):
            ids.add(p.stem)
        elif p.is_dir() and _dir_clips(p):
            ids.add(p.name)
    return sorted(ids)


def _candidates_hint(voices_dir: Path) -> str:
    cands = list_voices(voices_dir)
    return f"候補: {', '.join(cands)}" if cands else f"{voices_dir} に音声がありません"


def resolve_voice(voices_dir: Path, voice: str) -> list[Path]:
    voices_dir = Path(voices_dir)
    direct = Path(voice).expanduser()
    if direct.exists():
        if direct.is_dir():
            clips = _dir_clips(direct)
            if not clips:
                raise ValueError(f"ディレクトリに音声ファイルがありません: {direct}")
            return clips
        return [direct]

    files = [
        voices_dir / f"{voice}{ext}"
        for ext in sorted(AUDIO_EXTS)
        if (voices_dir / f"{voice}{ext}").is_file()
    ]
    d = voices_dir / voice
    clips = _dir_clips(d) if d.is_dir() else []
    if len(files) + (1 if clips else 0) > 1:
        raise ValueError(f"参照音声 {voice!r} が曖昧です（ファイルとディレクトリ、または拡張子違いが複数あります）")
    if files:
        return files
    if clips:
        return clips
    if d.is_dir():
        raise ValueError(f"ディレクトリに音声ファイルがありません: {d}")
    raise ValueError(f"参照音声 {voice!r} が見つかりません。{_candidates_hint(voices_dir)}")


def voice_key(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def describe_voice(paths: list[Path]) -> str:
    paths = [Path(p) for p in paths]
    name = paths[0].stem if len(paths) == 1 else paths[0].parent.name
    total, unknown = 0.0, False
    for p in paths:
        try:
            total += sf.info(str(p)).duration
        except Exception:  # soundfile が扱えない形式（mp3 など）や壊れたファイル
            unknown = True
    note = f"合計 {total:.1f} 秒"
    if unknown:
        note += "、秒数不明のクリップあり"
    return f"{name}（{len(paths)} クリップ、{note}）"
