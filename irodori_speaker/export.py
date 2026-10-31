from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from .assemble import safe_name

FORMATS = ("m4b", "mp3")
SOURCES = ("narration", "bgm")


@dataclass
class ExportInput:
    idx: int  # 0 始まり
    title: str  # chapters.json の生のタイトル
    wav: Path
    bgm_mp3: Path | None = None


@dataclass
class ExportChapter:
    idx: int
    title: str
    src: Path  # 実際にエンコードする入力（narration は章 WAV、bgm は章 MP3）
    sr: int
    frames: int  # src の frames
    sig: str  # src の .sig の中身


def chapter_title(idx: int, title: str) -> str:
    return title.strip() or f"第{idx + 1}章"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def collect_chapters(inputs: list[ExportInput], rows: list[dict], *, source: str) -> list[ExportChapter]:
    """入力を検証し、書き出す章の情報を作る。足りないものは ValueError で案内する。"""
    from .bgm_mix import TimesMismatch, load_times
    from .pipeline import _sig_path, _sounds_path

    if not inputs:
        raise ValueError("書き出す章がありません")
    paras: dict[int, set[int]] = {}
    for r in rows:
        paras.setdefault(int(r["chapter"]), set()).add(int(r["para"]))

    chapters: list[ExportChapter] = []
    for item in inputs:
        n = item.idx + 1
        hint = f"先に synth --chapters {n} を実行してください"
        if not item.wav.exists():
            raise ValueError(f"章 {n} の WAV がありません。{hint}（{item.wav}）")
        if not _sig_path(item.wav).exists():
            raise ValueError(f"章 {n} の WAV は合成が完了していません。{hint}")
        try:
            _sr, _frames, times = load_times(item.wav)
        except TimesMismatch as e:
            raise ValueError(f"章 {n} の WAV の時刻情報が合いません（{e}）。{hint}") from None
        if set(times) != paras.get(item.idx, set()):
            raise ValueError(f"章 {n} の WAV が今の script.tsv と段落が一致しません。{hint}")

        src = item.wav
        if source == "bgm":
            mp3 = item.bgm_mp3
            bgm_hint = f"先に bgm --chapters {n} を実行してください"
            if mp3 is None or not mp3.exists():
                raise ValueError(f"章 {n} の BGM 付き MP3 がありません。{bgm_hint}")
            if not _sig_path(mp3).exists() or not _sounds_path(mp3).exists():
                raise ValueError(f"章 {n} の BGM 付き MP3 は作成が完了していません。{bgm_hint}")
            src = mp3
        info = sf.info(str(src))
        chapters.append(
            ExportChapter(
                item.idx, item.title, src, info.samplerate, info.frames, _read(_sig_path(src))
            )
        )
    rates = {c.sr for c in chapters}
    if len(rates) > 1:
        raise ValueError(f"章ごとのサンプリングレートが一致しません: {sorted(rates)}")
    return chapters


def chapter_starts_ms(chapters: list[ExportChapter]) -> list[int]:
    """各章の開始ミリ秒と、最後に全体の終わり（章数 + 1 個）。累積 frames から丸めるのでずれが溜まらない。"""
    sr = chapters[0].sr
    bounds = [0]
    total = 0
    for c in chapters:
        total += c.frames
        bounds.append(round(total * 1000 / sr))
    return bounds


def escape_ffmeta(value: str) -> str:
    out = value.replace("\\", "\\\\")
    for ch in "=;#":
        out = out.replace(ch, "\\" + ch)
    return out.replace("\n", "\\\n")


def ffmetadata(book: str, chapters: list[ExportChapter]) -> str:
    bounds = chapter_starts_ms(chapters)
    lines = [";FFMETADATA1", f"title={escape_ffmeta(book)}", f"album={escape_ffmeta(book)}"]
    for i, c in enumerate(chapters):
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={bounds[i]}",
            f"END={bounds[i + 1]}",
            f"title={escape_ffmeta(chapter_title(c.idx, c.title))}",
        ]
    return "\n".join(lines) + "\n"


def export_paths(out_dir: Path, *, source: str, fmt: str) -> tuple[Path, Path | None]:
    """(通しの出力, 章ごとの MP3 を置くディレクトリ。m4b は None)。"""
    suffix = "_bgm" if source == "bgm" else ""
    base = out_dir / "export"
    full = base / f"{out_dir.name}{suffix}.{fmt}"
    return full, (base / f"chapters{suffix}" if fmt == "mp3" else None)


def chapter_mp3_path(chapter_dir: Path, ch: ExportChapter) -> Path:
    return chapter_dir / f"{ch.idx + 1:02d}_{safe_name(chapter_title(ch.idx, ch.title))}.mp3"


def export_signature(
    chapters: list[ExportChapter], *, source: str, fmt: str, bitrate: str, channels: int, book: str = ""
) -> str:
    payload = [
        [c.sig for c in chapters],
        [chapter_title(c.idx, c.title) for c in chapters],
        source,
        fmt,
        bitrate,
        channels,
        book,
    ]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()
