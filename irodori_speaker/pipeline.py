from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

from .assemble import Clip, concat_wavs, safe_name, write_chapter
from .config import EMOTIONS, Config
from .text_splitter import Chapter, build_chapters, read_text

TSV_FIELDS = ["chapter", "para", "seg", "kind", "labels", "text", "source"]

Progress = Callable[[int, int, str], None]


@dataclass
class BookResult:
    chapter_files: list[Path]
    full_path: Path | None
    stopped: bool
    stale_chapters: list[int] = field(default_factory=list)  # 1 始まり。台本と一致しない章 WAV


def parse_chapter_range(spec: str | None) -> set[int] | None:
    """"1-3,5" (1 始まり) -> {1,2,3,5}"""
    if not spec:
        return None
    result: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        result.update(range(int(lo), int(hi or lo) + 1))
    return result


def book_dir(cfg: Config, novel: Path) -> Path:
    return Path(cfg.output_dir) / novel.stem


def load_script(path: Path) -> dict[tuple[int, int, int], dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {
            (int(r["chapter"]), int(r["para"]), int(r["seg"])): r
            for r in csv.DictReader(f, delimiter="\t")
        }


def write_script(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def load_book(out_dir: Path) -> tuple[dict[int, str], list[dict]]:
    chapters_path = out_dir / "chapters.json"
    script_path = out_dir / "script.tsv"
    for p in (chapters_path, script_path):
        if not p.exists():
            raise ValueError(f"先に script を実行してください: {p}")
    data = json.loads(chapters_path.read_text(encoding="utf-8"))
    titles = {int(c["idx"]): c["title"] for c in data["chapters"]}
    rows = [r for _, r in sorted(load_script(script_path).items())]
    return titles, rows


def spoken_text(row: dict) -> str:
    return row["text"]


def build_script(
    novel: Path,
    cfg: Config,
    out_dir: Path,
    *,
    chapters: set[int] | None = None,
    use_emotion: bool = True,
    reannotate: bool = False,
    llm_overrides: dict | None = None,
    progress: Progress | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    built = build_chapters(read_text(novel), cfg.split.chapter_regex, cfg.split.max_chars)
    (out_dir / "chapters.json").write_text(
        json.dumps(
            {"chapters": [{"idx": c.idx, "title": c.title} for c in built]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    all_idx = {c.idx for c in built}
    if chapters is not None:
        built = [c for c in built if c.idx + 1 in chapters]
    if not built:
        raise ValueError("対象の章がありません")
    if llm_overrides:
        cfg = replace(cfg, llm=replace(cfg.llm, **llm_overrides))
    rows = annotate_chapters(built, cfg, out_dir, use_emotion, reannotate=reannotate, progress=progress)
    # 小説から削除された章の行は残さない（章の絞り込みの有無にかかわらず）
    script_path = out_dir / "script.tsv"
    existing = load_script(script_path)
    stale = [k for k in existing if k[0] not in all_idx]
    if stale:
        write_script(script_path, [r for k, r in sorted(existing.items()) if k[0] in all_idx])
    return rows


def annotate_chapters(
    chapters: list[Chapter],
    cfg: Config,
    out_dir: Path,
    use_emotion: bool,
    *,
    reannotate: bool = False,
    progress: Progress | None = None,
) -> list[dict]:
    """全セグメントの script.tsv の行を作る。手で直した script.tsv は優先する。"""
    script_path = out_dir / "script.tsv"
    overrides = {} if reannotate else load_script(script_path)
    use_llm = use_emotion and cfg.emotion_mode != "none"

    # 段落ごとに、保存済みの行を使えるかどうかを決める
    plan: list[tuple[list, list | None, list | None, bool]] = []
    for chapter in chapters:
        paras = chapter.paragraphs
        for p_i, segs in enumerate(paras):
            prev = paras[p_i - 1] if p_i > 0 else None
            nxt = paras[p_i + 1] if p_i + 1 < len(paras) else None
            kept = all(
                (r := overrides.get(s.key)) is not None and r["source"] == s.text for s in segs
            )
            plan.append((segs, prev, nxt, kept))
    total = len(plan)

    def saved_row(s) -> dict | None:
        # 保存済みで本文が一致する行は、段落の他の文が変わっていても手修正を保つ
        r = overrides.get(s.key)
        return _row(s, r["labels"], r["text"]) if r is not None and r["source"] == s.text else None

    def neutral_row(s) -> dict:
        return saved_row(s) or _row(s, "neutral:1", s.text)

    annotated: dict[int, list[dict]] = {}
    pending = [i for i, (_, _, _, kept) in enumerate(plan) if not kept]
    if use_llm and pending:
        from .emotion import (  # emotion が無い環境でも --no-emotion できるよう遅延 import
            LLMAnnotator,
            LLMServer,
            apply_shift,
            build_spoken_text,
            labels_to_str,
            split_clauses,
        )

        with LLMServer(cfg.llm):
            annotator = LLMAnnotator(cfg.llm, out_dir / "annotations.jsonl")
            try:
                results = annotator.annotate_many(
                    [plan[i][:3] for i in pending],
                    progress=lambda done, _total: progress
                    and progress(done, total, "LLM で感情を推定中 …"),
                )
            finally:
                annotator.unload()
        for i, para_anns in zip(pending, results):
            rows = []
            for s, sent_anns in zip(plan[i][0], para_anns):
                if (kept_row := saved_row(s)) is not None:
                    rows.append(kept_row)
                    continue
                anns = []
                for a in sent_anns:
                    a = apply_shift(a, s.kind, cfg.narration_intensity_shift)
                    if a.emotion not in EMOTIONS:
                        a = type(a)()
                    anns.append(a)
                text = build_spoken_text(split_clauses(s.text), anns, cfg.emotion_emoji)
                rows.append(_row(s, labels_to_str(anns), text))
            annotated[i] = rows

    # LLM の段落は annotate_many の進捗で数え済み。残りの段落をここで加算する
    counted = len(annotated)
    rows: list[dict] = []
    for i, (segs, _, _, kept) in enumerate(plan):
        if kept:
            rows.extend(_row(s, overrides[s.key]["labels"], overrides[s.key]["text"]) for s in segs)
        elif i in annotated:
            rows.extend(annotated[i])
        else:
            rows.extend(neutral_row(s) for s in segs)
        if progress and i not in annotated:
            counted += 1
            progress(counted, total, f"章 {segs[0].chapter_idx + 1} / 段落 {segs[0].para_idx + 1}")

    # --chapters で一部の章だけ処理した場合に、処理対象外の章の行（手修正を含む）を消さない。
    # 処理した章では今回存在しない行（小説から削除した文）を残さない
    processed = {c.idx for c in chapters}
    existing = load_script(script_path)
    others = [r for r in existing.values() if int(r["chapter"]) not in processed]
    merged = sorted(rows + others, key=lambda r: (int(r["chapter"]), int(r["para"]), int(r["seg"])))
    write_script(script_path, merged)
    return rows


def _row(s, labels: str, text: str) -> dict:
    return {
        "chapter": s.chapter_idx,
        "para": s.para_idx,
        "seg": s.seg_idx,
        "kind": s.kind,
        "labels": labels,
        "text": text,
        "source": s.text,
    }


def synthesize_book(
    rows: list[dict],
    titles: dict[int, str],
    cfg: Config,
    out_dir: Path,
    synth,
    *,
    chapters: set[int] | None = None,
    force: bool = False,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> BookResult:
    by_chapter: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_chapter[int(r["chapter"])].append(r)
    targets = [
        (idx, by_chapter[idx])
        for idx in sorted(by_chapter)
        if chapters is None or idx + 1 in chapters
    ]
    total = sum(len(rs) for _, rs in targets)
    done = 0
    chapter_files: list[Path] = []
    try:
        for idx, chapter_rows in targets:
            clips: list[Clip] = []
            for r in _sorted_rows(chapter_rows):
                if should_stop and should_stop():
                    _remove_old_full(out_dir)
                    return BookResult(chapter_files, None, stopped=True)
                path = synth.synthesize(spoken_text(r), force=force)
                clips.append(Clip(path, int(r["chapter"]), int(r["para"]), int(r["seg"])))
                done += 1
                if progress:
                    progress(done, total, f"章 {idx + 1} / 文 {done}/{total}")
            ch_path = _chapter_path(out_dir, idx, titles)
            # 書き込み失敗時に以前の署名が残って壊れた章が一致扱いにならないよう、先に消す
            _sig_path(ch_path).unlink(missing_ok=True)
            write_chapter(ch_path, clips, cfg.silence)
            _sig_path(ch_path).write_text(chapter_signature(_sorted_rows(by_chapter[idx]), cfg, synth), encoding="utf-8")
            chapter_files.append(ch_path)
    except Exception:
        # 台本と一致しない古い通しを残さない
        _remove_old_full(out_dir)
        raise

    # 全章の WAV があり、署名が現在の台本と一致するときだけ通しを作る
    all_paths: list[Path] = []
    stale: list[int] = []
    missing = False
    for idx in sorted(titles):
        p = _chapter_path(out_dir, idx, titles)
        all_paths.append(p)
        if not p.exists():
            missing = True
            continue
        sig = _sig_path(p)
        expected = chapter_signature(_sorted_rows(by_chapter.get(idx, [])), cfg, synth)
        if not sig.exists() or sig.read_text(encoding="utf-8").strip() != expected:
            stale.append(idx + 1)
            print(
                f"[pipeline] 章 {idx + 1} の音声が台本と一致しないため通しは作りません"
                f"（synth --chapters {idx + 1} で作り直してください）"
            )
    full: Path | None = None
    if all_paths and not missing and not stale:
        full = out_dir / f"{out_dir.name}_full.wav"
        try:
            concat_wavs(all_paths, full)
        except Exception:
            _remove_old_full(out_dir)
            raise
    else:
        _remove_old_full(out_dir)
    return BookResult(chapter_files, full, stopped=False, stale_chapters=stale)


def _remove_old_full(out_dir: Path) -> None:
    old = out_dir / f"{out_dir.name}_full.wav"
    if old.exists():
        old.unlink()
        print(f"[pipeline] 古い通し WAV を削除しました: {old}")


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (int(r["chapter"]), int(r["para"]), int(r["seg"])))


def _sig_path(chapter_wav: Path) -> Path:
    return chapter_wav.with_name(chapter_wav.name + ".sig")


def chapter_signature(rows: list[dict], cfg: Config, synth) -> str:
    payload = [
        [spoken_text(r) for r in rows],
        [(int(r["para"]), int(r["seg"])) for r in rows],
        asdict(cfg.sampling),
        asdict(cfg.silence),
        cfg.checkpoint,
        getattr(synth, "latent_key", None),
    ]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _chapter_path(out_dir: Path, idx: int, titles: dict[int, str]) -> Path:
    return out_dir / "chapters" / f"{idx + 1:02d}_{safe_name(titles.get(idx, ''))}.wav"
