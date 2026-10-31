from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

from .assemble import Clip, concat_wavs, safe_name, write_chapter
from .config import EMOTIONS, Config
from .synth import TEXT_RULE
from .text_splitter import Chapter, build_chapters, read_text

TSV_FIELDS = ["chapter", "para", "seg", "kind", "labels", "text", "source"]

Progress = Callable[[int, int, str], None]


@dataclass
class BookResult:
    chapter_files: list[Path]
    full_path: Path | None
    stopped: bool
    stale_chapters: list[int] = field(default_factory=list)  # 1 始まり。台本と一致しない章 WAV


@dataclass
class BgmResult:
    chapter_files: list[Path]
    full_path: Path | None
    missing: int  # missing.tsv の行数（見つからなかった音の枠と LLM が失敗した場面）


@dataclass
class BgmDeps:
    """build_bgm が外に頼る部品。None なら本物を作る（テストで偽物に差し替える）。"""

    client_factory: Callable[[str], object] | None = None  # (API キー) -> Freesound のクライアント
    planner_factory: Callable[[object, Path], object] | None = None  # (llm 設定, キャッシュ) -> ScenePlanner
    gate_factory: Callable[[object, Path], object] | None = None  # (llm 設定, キャッシュ) -> Gate
    scorer_factory: Callable[[object], object] | None = None  # (bgm 設定) -> ClapScorer
    server_factory: Callable[[object], AbstractContextManager] | None = None  # (llm 設定) -> LLMServer
    unload: Callable[[object], None] | None = None  # (llm 設定)
    download: Callable[[str, Path], None] | None = None  # (URL, 保存先)


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


def _sounds_path(mp3: Path) -> Path:
    return mp3.with_name(mp3.name + ".sounds.json")


def _write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_credits_from_mp3s(out_dir: Path) -> None:
    """CREDITS.txt を、実在する章 MP3 の .sounds.json にある音の和集合から作る。"""
    from . import bgm_mix
    from .bgm_types import Choice, Sound

    bgm_dir = out_dir / "bgm"
    if not (bgm_dir / "chapters").exists():
        return
    choices: list[Choice] = []
    # 章の題名の変更・削除で旧名のまま残った MP3 も含め、.sounds.json のある MP3 をすべて集計する
    for mp3 in sorted((bgm_dir / "chapters").glob("*.mp3")):
        sounds_file = _sounds_path(mp3)
        if not sounds_file.exists():
            continue
        try:
            sounds = [Sound.from_dict(d) for d in json.loads(sounds_file.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError, KeyError):
            continue
        choices.extend(Choice(0, 0, "ambient", "", snd, None, None) for snd in sounds)
    bgm_mix.write_credits(bgm_dir / "CREDITS.txt", choices)


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
        TEXT_RULE,
    ]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _chapter_path(out_dir: Path, idx: int, titles: dict[int, str]) -> Path:
    return out_dir / "chapters" / f"{idx + 1:02d}_{safe_name(titles.get(idx, ''))}.wav"


def build_bgm(
    cfg: Config,
    out_dir: Path,
    *,
    chapters: set[int] | None = None,
    replan: bool = False,
    llm_overrides: dict | None = None,
    deps: BgmDeps | None = None,
    progress: Progress | None = None,
) -> BgmResult:
    """合成済みの章 WAV に環境音と音楽を重ねて、章ごとと通しの MP3 を作る。"""
    try:
        return _build_bgm(
            cfg, out_dir, chapters=chapters, replan=replan, llm_overrides=llm_overrides,
            deps=deps, progress=progress,
        )
    except BaseException:
        # 事前検証で止まったときも含め、古い通しを残さない
        (out_dir / "bgm" / f"{out_dir.name}_bgm_full.mp3").unlink(missing_ok=True)
        raise
    finally:
        # 成功でも失敗でも、実在する章 MP3 で使っている音のクレジットを作り直す
        _write_credits_from_mp3s(out_dir)


def _build_bgm(
    cfg: Config,
    out_dir: Path,
    *,
    chapters: set[int] | None = None,
    replan: bool = False,
    llm_overrides: dict | None = None,
    deps: BgmDeps | None = None,
    progress: Progress | None = None,
) -> BgmResult:
    """合成済みの章 WAV に環境音と音楽を重ねて、章ごとと通しの MP3 を作る。"""
    # bgm_* と emotion は、Mac のテストと script が壊れないよう呼ばれたときに import する
    from . import audio_io, bgm_mix, bgm_scenes, bgm_select, freesound

    deps = deps or BgmDeps()
    if llm_overrides:
        cfg = replace(cfg, llm=replace(cfg.llm, **llm_overrides))
    bgm = cfg.bgm
    bgm_dir = out_dir / "bgm"
    cache_dir = bgm_dir / "cache"
    preview_dir = Path(cfg.output_dir) / ".freesound" / "previews"  # 本をまたぐ共有キャッシュ
    full_path = bgm_dir / f"{out_dir.name}_bgm_full.mp3"

    titles, rows = load_book(out_dir)
    all_chapters = {int(r["chapter"]) for r in rows}
    targets = sorted(i for i in all_chapters if chapters is None or i + 1 in chapters)
    if not targets:
        raise ValueError("対象の章がありません")

    # 1. 段落の時刻。無ければ LLM を動かす前に止める
    times: dict[int, tuple[int, int, dict[int, tuple[int, int]]]] = {}
    for idx in targets:
        wav = _chapter_path(out_dir, idx, titles)
        try:
            if not wav.exists():
                raise bgm_mix.TimesMismatch(f"{wav.name} がありません")
            times[idx] = bgm_mix.load_times(wav)
        except bgm_mix.TimesMismatch:
            raise ValueError(
                f"章 {idx + 1} の段落の時刻がありません。synth --chapters {idx + 1} を実行してください"
                "（文のキャッシュがあるので音声合成はやり直しません）"
            ) from None
        # 台本を更新して一部の章だけ synth し直した場合の食い違い（段落の並びだけを比べる。
        # 段落の中の文言だけの変更は検出できない。署名の完全な照合には synth の latent_key が要る）
        script_paras = sorted({int(r["para"]) for r in rows if int(r["chapter"]) == idx})
        if sorted(times[idx][2]) != script_paras:
            raise ValueError(
                f"章 {idx + 1} の朗読が今の台本と合いません。synth --chapters {idx + 1} を実行してください"
            )
        if not _sig_path(wav).exists():  # synth が途中で止まった章
            raise ValueError(
                f"章 {idx + 1} の朗読の署名がありません。synth --chapters {idx + 1} を実行してください"
                "（文のキャッシュがあるので音声合成はやり直しません）"
            )

    # 2. キー（LLM を起動する前に確かめる）
    key = freesound.load_key(bgm.secrets_file)

    # 3. 場面・検索・ふるい分け（LLM）
    if replan:
        (cache_dir / "scenes.jsonl").unlink(missing_ok=True)
    client = (deps.client_factory or freesound.FreesoundClient)(key)
    if deps.server_factory is None or deps.unload is None:
        from .emotion import LLMServer, unload_model
    server_factory = deps.server_factory or LLMServer
    unload = deps.unload or unload_model
    planner_factory = deps.planner_factory or (
        lambda llm, path: bgm_scenes.ScenePlanner(llm, path)
    )
    gate_factory = deps.gate_factory or (lambda llm, path: bgm_select.Gate(llm, path))

    paragraphs = bgm_scenes.chapter_paragraphs(rows)
    scenes: dict[int, list] = {}
    with server_factory(cfg.llm):
        try:
            planner = planner_factory(cfg.llm, cache_dir / "scenes.jsonl")
            for n, idx in enumerate(targets):
                if progress:
                    progress(n, len(targets), f"章 {idx + 1} の場面を区切り中 …")
                paras = paragraphs.get(idx, [])
                planned = planner.plan_chapter(idx, paras, bgm.scene_chunk_chars)
                scenes[idx] = bgm_scenes.merge_short(
                    planned, {p: len(t) for p, t in paras}, bgm.min_scene_chars
                )
            bgm_scenes.save_scenes(bgm_dir / "scenes.json", scenes, all_chapters)

            slots = bgm_select.build_slots(scenes, bgm)
            if progress:
                progress(0, len(slots), "Freesound を検索中 …")
            found, failures = bgm_select.search_slots(
                slots, client, cache_dir / "search.json", bgm.candidates
            )
            if progress:
                progress(0, len(slots), "候補をふるい分け中 …")
            gate = gate_factory(cfg.llm, cache_dir / "gate.jsonl")
            survivors, gate_rows, llm_error = bgm_select.gate_slots(
                slots, found, gate, cfg.llm.concurrency
            )
            bgm_select.write_gate_tsv(bgm_dir / "gate.tsv", gate_rows)
        finally:
            unload(cfg.llm)

    # 4. CLAP で順位を付けて選ぶ（LLM を解放した後）
    if progress:
        progress(0, len(slots), "CLAP で音を選んでいます …")
    scorer = (deps.scorer_factory or (lambda b: bgm_select.ClapScorer(b.clap_model, b.device)))(bgm)
    try:
        choices = bgm_select.choose(
            slots, found, survivors, {**failures, **llm_error}, scorer,
            cache_dir / "scores.json", bgm_select.read_exclude(bgm_dir / "exclude.txt"),
            preview_dir, bgm.download_workers,
            **({"download": deps.download} if deps.download else {}),
        )
    finally:
        del scorer
    bgm_select.write_selection(bgm_dir / "selection.json", choices, set(targets), all_chapters)
    # 他の章の行を消さないよう、保存済みの全章分（selection.json / scenes.json）から作る
    bgm_select.write_missing(
        bgm_dir / "missing.tsv",
        bgm_select.load_selection(bgm_dir / "selection.json"),
        bgm_scenes.load_scenes(bgm_dir / "scenes.json"),
    )

    # 5. 章ごとの重ね合わせ
    download = deps.download or freesound.download_preview
    chapter_files: list[Path] = []
    signatures: dict[int, str] = {}  # 今回の実行で確かめた章の署名
    for n, idx in enumerate(targets):
        if progress:
            progress(n, len(targets), f"章 {idx + 1} に音を重ねています …")
        wav = _chapter_path(out_dir, idx, titles)
        mp3 = bgm_dir / "chapters" / f"{wav.stem}.mp3"
        sr, frames, paras_t = times[idx]
        chapter_choices = {
            f"{c.chapter}:{c.scene}:{c.kind}": c for c in choices if c.chapter == idx
        }
        sig_file = _sig_path(mp3)
        sig_src = _sig_path(wav)
        narration_sig = sig_src.read_text(encoding="utf-8").strip()
        signature = bgm_mix.mix_signature(narration_sig, scenes[idx], chapter_choices, bgm)
        chapter_files.append(mp3)
        signatures[idx] = signature
        if (
            mp3.exists()
            and sig_file.exists()
            and sig_file.read_text(encoding="utf-8").strip() == signature
            and _sounds_path(mp3).exists()
        ):
            continue
        sig_file.unlink(missing_ok=True)  # 書き込みに失敗したら stale 扱いにする
        try:
            layout = bgm_mix.build_layout(sr, frames, paras_t, scenes[idx], chapter_choices, bgm)
            sounds = {}
            used: dict[int, object] = {}  # この MP3 で実際に使った音（初めて使った順）
            decoded: dict[int, object] = {}  # 同じ音は章の中で 1 回だけ読む（render は素材を書き換えない）
            for track in [*layout.ambient, *layout.music]:
                sound = chapter_choices[track.key].sound
                if sound.id not in decoded:
                    try:
                        dst = freesound.cached_preview(sound, preview_dir, download)
                    except (freesound.FreesoundError, OSError) as e:
                        raise RuntimeError(
                            f"章 {idx + 1} で使う音をダウンロードできませんでした（ID {sound.id}）: {e}"
                        ) from None
                    try:
                        decoded[sound.id] = audio_io.decode(dst, sr, 2)
                    except Exception:
                        dst.unlink(missing_ok=True)  # 読めないキャッシュは消し、次の実行で取り直す
                        raise
                    used[sound.id] = sound
                sounds[track.key] = decoded[sound.id]
            import soundfile as sf

            narration, _ = sf.read(wav, dtype="float32")
            if narration.ndim > 1:
                narration = narration[:, 0]
            audio = bgm_mix.render(layout, narration, sounds, sr, bgm)
            audio_io.write_mp3(audio, sr, mp3, bgm.mp3_bitrate)
            _write_json_atomic(_sounds_path(mp3), [x.to_dict() for x in used.values()])
            sig_file.write_text(signature, encoding="utf-8")
        except BaseException:
            # 古い音の MP3 を残さない（CREDITS.txt と食い違うため）
            for p in (mp3, sig_file, _sounds_path(mp3)):
                p.unlink(missing_ok=True)
            raise

    # 6. 通し
    # 通しは、章タイトルにある全章の MP3 が今回の署名と一致するときだけ（rows に無い章があれば作らない）
    if (
        chapters is None
        and titles
        and all(
            idx in signatures
            and (mp3 := bgm_dir / "chapters" / f"{_chapter_path(out_dir, idx, titles).stem}.mp3").exists()
            and _sig_path(mp3).exists()
            and _sig_path(mp3).read_text(encoding="utf-8").strip() == signatures[idx]
            for idx in titles
        )
    ):
        chapter_files_all = [
            bgm_dir / "chapters" / f"{_chapter_path(out_dir, idx, titles).stem}.mp3" for idx in sorted(titles)
        ]
        audio_io.concat_mp3(chapter_files_all, full_path)
        full: Path | None = full_path
    else:
        full_path.unlink(missing_ok=True)
        full = None

    missing_tsv = bgm_dir / "missing.tsv"
    missing = max(0, len(missing_tsv.read_text(encoding="utf-8").splitlines()) - 1) if missing_tsv.exists() else 0
    return BgmResult(chapter_files, full, missing)


@dataclass
class ExportResult:
    path: Path  # 通しの出力（m4b、または mp3 の通し）
    chapter_files: list[Path]  # mp3 のときの章ごとのファイル（m4b は空）
    skipped: bool  # 署名が一致して再エンコードしなかった


def _remove_export_outputs(full: Path, chapter_dir: Path | None) -> None:
    """通し・.sig・章ディレクトリ（export 専用）の *.mp3 をすべて消す。"""
    stale = [full, _sig_path(full)]
    if chapter_dir is not None and chapter_dir.is_dir():
        stale += chapter_dir.glob("*.mp3")
    for p in stale:
        p.unlink(missing_ok=True)


def build_export(
    cfg: Config,
    out_dir: Path,
    *,
    source: str = "narration",
    fmt: str | None = None,
    bitrate: str | None = None,
    progress: Progress | None = None,
) -> ExportResult:
    """合成済みの章（source="bgm" なら BGM 付きの章 MP3）を、聴く形式（m4b / mp3）に書き出す。"""
    from . import audio_io, export

    fmt = fmt or cfg.export.format
    bitrate = bitrate or cfg.export.bitrate
    if fmt not in export.FORMATS:
        raise ValueError(f"形式は {' / '.join(export.FORMATS)} のどちらかにしてください: {fmt}")
    if source not in export.SOURCES:
        raise ValueError(f"source は {' / '.join(export.SOURCES)} のどちらかにしてください: {source}")
    full, chapter_dir = export.export_paths(out_dir, source=source, fmt=fmt)
    try:
        titles, rows = load_book(out_dir)
        inputs = []
        for idx in sorted(titles):
            wav = _chapter_path(out_dir, idx, titles)
            mp3 = out_dir / "bgm" / "chapters" / f"{wav.stem}.mp3" if source == "bgm" else None
            inputs.append(export.ExportInput(idx, titles[idx], wav, mp3))
        chapters = export.collect_chapters(inputs, rows, source=source)
    except BaseException:
        # 今の本と合わない古い書き出しを使えないよう、今回の source / fmt の出力だけ消す
        _remove_export_outputs(full, chapter_dir)
        raise

    book = cfg.export.title or out_dir.name
    channels = 2 if source == "bgm" else 1
    chapter_paths = [export.chapter_mp3_path(chapter_dir, c) for c in chapters] if chapter_dir else []
    signature = export.export_signature(
        chapters, source=source, fmt=fmt, bitrate=bitrate, channels=channels, book=book
    )
    sig_file = _sig_path(full)
    if (
        full.exists()
        and all(p.exists() for p in chapter_paths)
        and sig_file.exists()
        and sig_file.read_text(encoding="utf-8").strip() == signature
    ):
        return ExportResult(full, chapter_paths, True)

    steps = len(chapters) + 1 if fmt == "mp3" else 1
    done = 0

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, steps, msg)

    if progress:
        progress(0, steps, "書き出し中")
    sig_file.unlink(missing_ok=True)
    try:
        if fmt == "m4b":
            audio_io.write_m4b(
                [c.src for c in chapters], export.ffmetadata(book, chapters), full, bitrate, channels
            )
            tick("書き出し完了")
        else:
            for n, (c, path) in enumerate(zip(chapters, chapter_paths), start=1):
                title = export.chapter_title(c.idx, c.title)
                audio_io.encode_file(
                    c.src, path,
                    [
                        "-c:a", "libmp3lame", "-b:a", bitrate, "-ac", str(channels),
                        "-metadata", f"title={title}", "-metadata", f"album={book}",
                        "-metadata", f"track={c.idx + 1}", "-f", "mp3",
                    ],
                )
                tick(f"章 {n}/{len(chapters)}")
            audio_io.concat_mp3(chapter_paths, full)
            tick("通しを連結")
        if export.input_stamps(chapters) != [(c.stamp, c.sig) for c in chapters]:
            raise RuntimeError("書き出し中に章の音声が更新されました。もう一度書き出してください")
        sig_file.write_text(signature, encoding="utf-8")
        if chapter_dir is not None:
            # 章題の変更・章の削除で残った旧い章 MP3（chapter_dir は export 専用）
            for p in chapter_dir.glob("*.mp3"):
                if p not in chapter_paths:
                    p.unlink(missing_ok=True)
    except BaseException:
        # 新旧の章が混ざらないよう、旧名のものも含めて章 MP3 をすべて消す
        _remove_export_outputs(full, chapter_dir)
        raise
    return ExportResult(full, chapter_paths, False)
