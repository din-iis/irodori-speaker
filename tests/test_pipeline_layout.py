"""場面区切り・章題の扱い（script.tsv の行・手直しの引き継ぎ・合成・時刻）。"""

from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import bgm_mix, cli, pipeline
from irodori_speaker.config import Config


@pytest.fixture
def fake_emotion(monkeypatch):
    calls: list[list] = []
    mod = types.ModuleType("irodori_speaker.emotion")

    @dataclass
    class Annotation:
        emotion: str = "neutral"
        intensity: int = 1

    class LLMAnnotator:
        def __init__(self, cfg, cache_path):
            pass

        def annotate_many(self, items, progress=None):
            calls.append(items)
            return [[[Annotation()] for _ in segs] for segs, _p, _n in items]

        def unload(self):
            pass

    class LLMServer:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    for name, obj in {
        "split_clauses": lambda s: [s],
        "Annotation": Annotation,
        "apply_shift": lambda a, kind, shift: a,
        "build_spoken_text": lambda clauses, anns, table=None: "".join(clauses),
        "labels_to_str": lambda anns: "|".join(f"{a.emotion}:{a.intensity}" for a in anns),
        "LLMAnnotator": LLMAnnotator,
        "LLMServer": LLMServer,
    }.items():
        setattr(mod, name, obj)
    monkeypatch.setitem(sys.modules, "irodori_speaker.emotion", mod)
    return calls


class FakeSynth:
    def __init__(self, segments_dir: Path) -> None:
        self.segments_dir = segments_dir
        self.texts: list[str] = []

    def synthesize(self, text: str, force: bool = False, seed: int | None = None) -> Path:
        self.texts.append(text)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        path = self.segments_dir / f"{hashlib.sha1(text.encode()).hexdigest()}.wav"
        sf.write(path, np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_16")
        return path


def _cfg(tmp_path: Path, **kw) -> Config:
    return Config(output_dir=str(tmp_path / "out"), **kw)


def _novel(tmp_path: Path, text: str, name: str = "n.txt") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _build(tmp_path, text, cfg=None, **kw):
    out = tmp_path / "book"
    out.mkdir(exist_ok=True)
    return out, pipeline.build_script(_novel(tmp_path, text), cfg or _cfg(tmp_path), out, **kw)


def _edit(out: Path, key, text: str) -> None:
    rows = pipeline.load_script(out / "script.tsv")
    rows[key]["text"] = text
    pipeline.write_script(out / "script.tsv", [r for _, r in sorted(rows.items())])


BASIC = "第一章 題\nあい。\n＊\nうえ。\nおか。\n"


def test_break_and_title_rows(tmp_path):
    _, rows = _build(tmp_path, BASIC, use_emotion=False)
    by = {(r["para"], r["seg"]): r for r in rows}
    assert by[(0, 0)]["kind"] == "title"
    assert by[(0, 0)]["labels"] == "neutral:1" and by[(0, 0)]["text"] == "第一章 題"
    assert by[(2, 0)]["kind"] == "break"
    assert by[(2, 0)]["labels"] == "" and by[(2, 0)]["text"] == "" and by[(2, 0)]["source"] == "＊"


def test_break_text_is_always_empty_even_with_override(tmp_path):
    out, _ = _build(tmp_path, BASIC, use_emotion=False)
    _edit(out, (0, 2, 0), "手直し")
    rows = pipeline.build_script(_novel(tmp_path, BASIC), _cfg(tmp_path), out, use_emotion=False)
    assert {(r["para"]): r["text"] for r in rows}[2] == ""


def test_llm_never_sees_break_or_title_nor_as_context(tmp_path, fake_emotion):
    _build(tmp_path, BASIC)
    (items,) = fake_emotion
    shape = [([s.text for s in segs], [s.text for s in p] if p else None, [s.text for s in n] if n else None)
             for segs, p, n in items]
    assert shape == [
        (["あい。"], None, ["うえ。"]),
        (["うえ。"], ["あい。"], ["おか。"]),
        (["おか。"], ["うえ。"], None),
    ]


def test_title_edit_is_kept_without_llm(tmp_path, fake_emotion):
    out, _ = _build(tmp_path, BASIC)
    _edit(out, (0, 0, 0), "だいいちしょう")
    fake_emotion.clear()
    pipeline.build_script(_novel(tmp_path, BASIC), _cfg(tmp_path), out)
    assert pipeline.load_script(out / "script.tsv")[(0, 0, 0)]["text"] == "だいいちしょう"
    assert fake_emotion == []


def test_edits_follow_when_a_break_is_inserted(tmp_path, fake_emotion):
    out, _ = _build(tmp_path, BASIC)
    _edit(out, (0, 3, 0), "うえ手直し")
    fake_emotion.clear()
    inserted = "第一章 題\n零。\nあい。\n＊\nうえ。\nおか。\n"  # 先頭に 1 段落足して 1 つずれる
    pipeline.build_script(_novel(tmp_path, inserted), _cfg(tmp_path), out)
    rows = pipeline.load_script(out / "script.tsv")
    assert rows[(0, 4, 0)]["text"] == "うえ手直し"
    assert [s.text for call in fake_emotion for segs, _p, _n in call for s in segs] == ["零。"]


def test_same_sentence_edits_map_in_document_order(tmp_path):
    cfg = _cfg(tmp_path, read_chapter_title=False)
    out, _ = _build(tmp_path, "「はい」\nあ。\n「はい」\n", cfg, use_emotion=False)
    _edit(out, (0, 0, 0), "X")
    _edit(out, (0, 2, 0), "Y")
    pipeline.build_script(_novel(tmp_path, "零。\n「はい」\nあ。\n「はい」\n"), cfg, out, use_emotion=False)
    rows = pipeline.load_script(out / "script.tsv")
    assert rows[(0, 1, 0)]["text"] == "X"
    assert rows[(0, 3, 0)]["text"] == "Y"


def test_key_match_is_not_stolen_by_earlier_fallback(tmp_path):
    cfg = _cfg(tmp_path, read_chapter_title=False)
    out, _ = _build(tmp_path, "あ。\nい。\n「はい」\n", cfg, use_emotion=False)
    _edit(out, (0, 2, 0), "Y")
    pipeline.build_script(_novel(tmp_path, "「はい」\nい。\n「はい」\n"), cfg, out, use_emotion=False)
    rows = pipeline.load_script(out / "script.tsv")
    assert rows[(0, 2, 0)]["text"] == "Y"
    assert rows[(0, 0, 0)]["text"] == "「はい」"


def test_reannotate_drops_fallback_edits(tmp_path):
    out, _ = _build(tmp_path, BASIC, use_emotion=False)
    _edit(out, (0, 3, 0), "うえ手直し")
    shifted = "第一章 題\n零。\nあい。\n＊\nうえ。\nおか。\n"
    pipeline.build_script(_novel(tmp_path, shifted), _cfg(tmp_path), out, use_emotion=False, reannotate=True)
    assert pipeline.load_script(out / "script.tsv")[(0, 4, 0)]["text"] == "うえ。"


TWO = "第一章 A\nあい。\n＊\nうえ。\n第二章 B\nおか。\n◇\nきく。\n"


def test_chapter_subset_keeps_other_chapters_break_and_title(tmp_path):
    out, _ = _build(tmp_path, TWO, use_emotion=False)
    pipeline.build_script(_novel(tmp_path, TWO), _cfg(tmp_path), out, use_emotion=False, chapters={1})
    rows = pipeline.load_script(out / "script.tsv")
    assert {r["kind"] for k, r in rows.items() if k[0] == 1} >= {"title", "break"}
    assert {r["kind"] for k, r in rows.items() if k[0] == 0} >= {"title", "break"}


def test_vanished_break_and_sentence_rows_are_dropped(tmp_path):
    out, _ = _build(tmp_path, BASIC, use_emotion=False)
    pipeline.build_script(_novel(tmp_path, "第一章 題\nあい。\nうえ。\n"), _cfg(tmp_path), out, use_emotion=False)
    rows = pipeline.load_script(out / "script.tsv")
    assert [r["source"] for _, r in sorted(rows.items())] == ["第一章 題", "あい。", "うえ。"]


def _book(tmp_path, text=BASIC):
    out, _ = _build(tmp_path, text, use_emotion=False)
    titles, rows = pipeline.load_book(out)
    return out, titles, rows


def test_synthesize_book_skips_break_even_with_text(tmp_path):
    out, titles, rows = _book(tmp_path)
    rows = [dict(r, text="読まれない") if r["kind"] == "break" else r for r in rows]
    synth = FakeSynth(out / "segments")
    pipeline.synthesize_book(rows, titles, _cfg(tmp_path), out, synth)
    assert synth.texts == ["第一章 題", "あい。", "うえ。", "おか。"]


def test_progress_total_excludes_break(tmp_path):
    out, titles, rows = _book(tmp_path)
    seen: list[tuple[int, int]] = []
    pipeline.synthesize_book(rows, titles, _cfg(tmp_path), out, FakeSynth(out / "segments"),
                             progress=lambda d, t, m: seen.append((d, t)))
    assert seen[-1] == (4, 4)


def test_times_paragraphs_match_script_paragraphs(tmp_path):
    out, titles, rows = _book(tmp_path)
    res = pipeline.synthesize_book(rows, titles, _cfg(tmp_path), out, FakeSynth(out / "segments"))
    _sr, _frames, paras = bgm_mix.load_times(res.chapter_files[0])
    assert sorted(paras) == sorted({int(r["para"]) for r in rows})
    assert paras[2][0] == paras[2][1]  # 区切りは幅 0


def test_cli_script_honours_read_chapter_title_false(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"output_dir: {tmp_path / 'out'}\nread_chapter_title: false\n", encoding="utf-8")
    novel = _novel(tmp_path, BASIC, "nv.txt")
    assert cli.main(["script", str(novel), "-c", str(cfg), "--no-emotion"]) == 0
    rows = pipeline.load_script(tmp_path / "out" / "nv" / "script.tsv")
    assert "title" not in {r["kind"] for r in rows.values()}


def test_chapter_signature_changes_with_kind_only():
    rows = [{"chapter": 0, "para": 0, "seg": 0, "kind": "narration", "text": "あ"}]
    other = [dict(rows[0], kind="title")]
    cfg = Config()
    assert pipeline.chapter_signature(rows, cfg, None) != pipeline.chapter_signature(other, cfg, None)


def test_match_saved_rows_is_fast_when_everything_shifts():
    import time

    from irodori_speaker.text_splitter import Chapter, Segment

    n = 6000
    old = {(0, i, 0): {"source": f"文{i}。", "labels": "", "text": f"文{i}。"} for i in range(n)}
    # 題の挿入で全段落が 1 つずれる
    paras = [[Segment(0, 0, 0, "title", "題")]] + [
        [Segment(0, i + 1, 0, "narration", f"文{i}。")] for i in range(n)
    ]
    start = time.perf_counter()
    got = pipeline._match_saved_rows([Chapter(0, "題", paras)], old)
    assert time.perf_counter() - start < 1.0
    assert len(got) == n and got[(0, 5, 0)]["source"] == "文4。"
