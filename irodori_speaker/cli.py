from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from .config import Config, load_config
from .pipeline import (
    BookResult,
    book_dir,
    build_script,
    load_book,
    parse_chapter_range,
    synthesize_book,
)
from .text_splitter import build_chapters, read_text

TTS_MISSING = (
    "音声合成には torch と irodori_tts が必要です。"
    "GPU マシンで `uv sync --frozen --extra tts --extra cu128` を実行してください（{err}）"
)


def _make_synthesizer(cfg: Config, out_dir: Path, voice: str, checkpoint: str | None):
    # torch / irodori_tts が無い環境でも script だけは動くよう遅延 import
    from .synth import Synthesizer
    from .voices import describe_voice, resolve_voice

    if checkpoint:
        cfg = replace(cfg, checkpoint=checkpoint)
    voice_paths = resolve_voice(Path(cfg.voices_dir), voice)
    print(f"使う声: {describe_voice(voice_paths)}")
    return Synthesizer(
        cfg,
        out_dir / "segments",
        voice_paths,
        Path(cfg.output_dir) / ".voices",
        out_dir / "warnings.log",
    )


def _run_script(args, cfg: Config, out_dir: Path) -> int:
    chapters = build_chapters(read_text(args.novel), cfg.split.chapter_regex, cfg.split.max_chars)
    selected = parse_chapter_range(args.chapters)
    if selected is not None:
        chapters = [c for c in chapters if c.idx + 1 in selected]
    if not chapters:
        print("対象の章がありません")
        return 1
    n_segs = sum(len(p) for c in chapters for p in c.paragraphs)
    n_paras = sum(len(c.paragraphs) for c in chapters)
    print(f"{len(chapters)} 章 / {n_paras} 段落 / {n_segs} 文")

    overrides = {}
    if args.llm_model:
        overrides["model"] = args.llm_model
    if args.think:
        overrides["think"] = True
    with tqdm(total=n_paras, desc="台本作成", unit="段落") as bar:
        last = [0]

        def progress(done: int, total: int, msg: str) -> None:
            bar.update(done - last[0])
            last[0] = done

        build_script(
            args.novel,
            cfg,
            out_dir,
            chapters=selected,
            use_emotion=not args.no_emotion,
            reannotate=args.reannotate,
            llm_overrides=overrides or None,
            progress=progress,
        )
    print(f"script.tsv を出力しました: {out_dir / 'script.tsv'}")
    return 0


def _run_synth(args, cfg: Config, out_dir: Path) -> int:
    try:
        titles, rows = load_book(out_dir)
    except ValueError as e:
        print(e)
        return 1
    voice = args.voice or cfg.voice
    if not voice:
        print("config の voice か --voice で声を指定してください")
        return 1
    selected = parse_chapter_range(args.chapters)
    try:
        synth = _make_synthesizer(cfg, out_dir, voice, args.checkpoint)
    except ImportError as e:
        print(TTS_MISSING.format(err=e))
        return 2
    except ValueError as e:
        print(e)
        return 1

    with tqdm(total=1, desc="音声合成", unit="文") as bar:
        last = [0]

        def progress(done: int, total: int, msg: str) -> None:
            bar.total = total
            bar.update(done - last[0])
            last[0] = done

        try:
            result: BookResult = synthesize_book(
                rows, titles, cfg, out_dir, synth, chapters=selected, force=args.force, progress=progress
            )
        except ImportError as e:
            print(TTS_MISSING.format(err=e))
            return 2
    if not result.chapter_files:
        print("対象の章がありません")
        return 1
    if result.full_path:
        print(f"完了: {result.full_path}")
    else:
        print("章ごとの WAV を出力しました（全章が揃っていないため通しは未作成）:")
        for p in result.chapter_files:
            print(f"  {p}")
    warnings = out_dir / "warnings.log"
    n_warn = len(warnings.read_text(encoding="utf-8").splitlines()) if warnings.exists() else 0
    print(f"警告: {n_warn} 件（{warnings}）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="irodori_speaker", description="小説を Irodori-TTS で読み上げる")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("novel", type=Path, help="小説のテキストファイル")
        p.add_argument("-c", "--config", type=Path, help="設定 YAML")
        p.add_argument("--chapters", help="対象の章 (1 始まり) 例: 1-3,5")

    def script_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-emotion", action="store_true", help="LLM を使わず全て neutral にする")
        p.add_argument("--reannotate", action="store_true", help="手直しを捨てて感情推定をやり直す")
        p.add_argument("--llm-model", help="LLM のモデル名（config を上書き）")
        p.add_argument("--think", action="store_true", help="LLM の思考モードを有効にする")

    def synth_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--force", action="store_true", help="合成済みのセグメントも作り直す")
        p.add_argument("--voice", help="声の名前（config の voice を上書き）")
        p.add_argument("--checkpoint", help="TTS のチェックポイント（config を上書き）")

    for name, helptext, opts in (
        ("script", "台本 script.tsv を作る", [script_opts]),
        ("synth", "script.tsv から音声を作る", [synth_opts]),
        ("run", "台本作成と音声合成を続けて行う", [script_opts, synth_opts]),
    ):
        p = sub.add_parser(name, help=helptext)
        common(p)
        for o in opts:
            o(p)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = book_dir(cfg, args.novel)

    if args.command in ("script", "run"):
        rc = _run_script(args, cfg, out_dir)
        if rc != 0:
            return rc
    if args.command in ("synth", "run"):
        return _run_synth(args, cfg, out_dir)
    return 0
