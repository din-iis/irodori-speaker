from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import cli, pipeline
from irodori_speaker.config import Config
from irodori_speaker.text_splitter import build_chapters, read_text

NOVEL = Path(__file__).parent / "sample_novel.txt"


# ---------- 偽 emotion モジュール ----------


class Recorder:
    def __init__(self) -> None:
        self.annotate_calls: list[list] = []
        self.llm_models: list[str] = []
        self.server_entered = 0
        self.server_exited = 0
        self.unloaded = 0

    @property
    def annotated_texts(self) -> list[str]:
        return [s.text for call in self.annotate_calls for item in call for s in item[0]]


@pytest.fixture
def fake_emotion(monkeypatch):
    rec = Recorder()
    mod = types.ModuleType("irodori_speaker.emotion")

    def split_clauses(sentence: str) -> list[str]:
        out, buf = [], ""
        for ch in sentence:
            buf += ch
            if ch in "、。！？":
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
        return out

    @dataclass
    class Annotation:
        emotion: str = "neutral"
        intensity: int = 1

    def apply_shift(ann, kind, narration_shift):
        return ann

    def build_spoken_text(clauses, anns, table=None):
        return "".join(("😊" if a.emotion == "happy" else "") + c for c, a in zip(clauses, anns))

    def labels_to_str(anns):
        return "|".join(f"{a.emotion}:{a.intensity}" for a in anns)

    class LLMAnnotator:
        def __init__(self, cfg, cache_path):
            rec.llm_models.append(cfg.model)

        def annotate_many(self, items):
            rec.annotate_calls.append(items)
            result = []
            for segs, _prev, _nxt in items:
                para = []
                for s in segs:
                    a = Annotation("happy", 2) if s.kind == "dialogue" else Annotation("neutral", 1)
                    para.append([a for _ in split_clauses(s.text)])
                result.append(para)
            return result

        def unload(self):
            rec.unloaded += 1

    class LLMServer:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            rec.server_entered += 1
            return self

        def __exit__(self, *exc):
            rec.server_exited += 1

    for obj in (split_clauses, Annotation, apply_shift, build_spoken_text, labels_to_str,
                LLMAnnotator, LLMServer):
        setattr(mod, obj.__name__, obj)
    monkeypatch.setitem(sys.modules, "irodori_speaker.emotion", mod)
    return rec


# ---------- 偽 synth ----------


class FakeSynth:
    def __init__(self, segments_dir: Path) -> None:
        self.segments_dir = segments_dir
        self.texts: list[str] = []
        self.forces: list[bool] = []
        self.last_warnings: list[str] = []

    def synthesize(self, text: str, force: bool = False) -> Path:
        self.texts.append(text)
        self.forces.append(force)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        path = self.segments_dir / f"{hashlib.sha1(text.encode()).hexdigest()}.wav"
        sf.write(path, np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_16")
        return path


def make_cfg(tmp_path: Path) -> Config:
    return Config(output_dir=str(tmp_path / "out"))


def write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(f"output_dir: {tmp_path / 'out'}\nvoice: narrator\n", encoding="utf-8")
    return p


# ---------- parse_chapter_range ----------


def test_parse_chapter_range():
    assert pipeline.parse_chapter_range(None) is None
    assert pipeline.parse_chapter_range("") is None
    assert pipeline.parse_chapter_range("1-3,5") == {1, 2, 3, 5}
    assert pipeline.parse_chapter_range("2") == {2}


def test_book_dir(tmp_path):
    assert pipeline.book_dir(make_cfg(tmp_path), Path("x/novel.txt")) == tmp_path / "out" / "novel"


# ---------- build_script ----------


def test_build_script_writes_chapters_json_and_tsv(tmp_path, fake_emotion):
    import json

    out = tmp_path / "book"
    out.mkdir()
    rows = pipeline.build_script(NOVEL, make_cfg(tmp_path), out)
    data = json.loads((out / "chapters.json").read_text(encoding="utf-8"))
    assert [c["idx"] for c in data["chapters"]] == [0, 1]
    assert data["chapters"][0]["title"].startswith("第一章")
    header = (out / "script.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split("\t") == pipeline.TSV_FIELDS
    dialogue = [r for r in rows if r["kind"] == "dialogue"]
    assert dialogue and all("😊" in r["text"] for r in dialogue)
    assert all(r["text"] == r["source"] for r in rows if r["kind"] == "narration")


def test_build_script_no_emotion_does_not_call_llm(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    rows = pipeline.build_script(NOVEL, make_cfg(tmp_path), out, use_emotion=False)
    assert fake_emotion.annotate_calls == []
    assert fake_emotion.server_entered == 0
    assert rows and all(r["text"] == r["source"] and r["labels"] for r in rows)


def test_no_emotion_does_not_import_emotion(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "irodori_speaker.emotion", None)  # import すると ImportError
    out = tmp_path / "book"
    out.mkdir()
    rows = pipeline.build_script(NOVEL, make_cfg(tmp_path), out, use_emotion=False)
    assert rows


def test_emotion_mode_none_skips_llm(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = Config(output_dir=str(tmp_path / "out"), emotion_mode="none")
    rows = pipeline.build_script(NOVEL, cfg, out)
    assert fake_emotion.annotate_calls == []
    assert all(r["text"] == r["source"] for r in rows)


def test_manual_edit_is_kept(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    pipeline.build_script(NOVEL, cfg, out)
    first_calls = len(fake_emotion.annotate_calls)
    script = out / "script.tsv"
    rows = pipeline.load_script(script)
    key = next(k for k, r in rows.items() if r["kind"] == "dialogue")
    rows[key]["text"] = "手直し済み"
    pipeline.write_script(script, sorted(rows.values(), key=lambda r: tuple(int(r[f]) for f in ("chapter", "para", "seg"))))

    pipeline.build_script(NOVEL, cfg, out)
    assert len(fake_emotion.annotate_calls) == first_calls  # 呼ばれない
    assert pipeline.load_script(script)[key]["text"] == "手直し済み"


def test_only_changed_paragraph_is_reannotated(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    pipeline.build_script(NOVEL, cfg, out)
    fake_emotion.annotate_calls.clear()

    changed = tmp_path / "changed.txt"
    changed.write_text(read_text(NOVEL).replace("おはよう。今日は早いね", "こんにちは。今日は遅いね"), encoding="utf-8")
    pipeline.build_script(changed, cfg, out)
    texts = fake_emotion.annotated_texts
    assert any("こんにちは" in t for t in texts)
    assert not any("眠れなかった" in t for t in texts)
    assert not any("雨が降っていた" in t for t in texts)


def test_reannotate_discards_manual_edit(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    pipeline.build_script(NOVEL, cfg, out)
    script = out / "script.tsv"
    rows = pipeline.load_script(script)
    key = next(k for k, r in rows.items() if r["kind"] == "dialogue")
    original = rows[key]["text"]
    rows[key]["text"] = "手直し済み"
    pipeline.write_script(script, list(rows.values()))

    pipeline.build_script(NOVEL, cfg, out, reannotate=True)
    assert pipeline.load_script(script)[key]["text"] == original


def test_chapter_subset_keeps_other_chapters(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    pipeline.build_script(NOVEL, cfg, out, use_emotion=False)
    rows = pipeline.build_script(NOVEL, cfg, out, chapters={2}, use_emotion=False)
    assert {int(r["chapter"]) for r in rows} == {1}
    saved = pipeline.load_script(out / "script.tsv")
    assert {k[0] for k in saved} == {0, 1}


def test_build_script_no_matching_chapter(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    with pytest.raises(ValueError):
        pipeline.build_script(NOVEL, make_cfg(tmp_path), out, chapters={9})


def test_llm_overrides_reach_annotator(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    pipeline.build_script(NOVEL, make_cfg(tmp_path), out, llm_overrides={"model": "x"})
    assert fake_emotion.llm_models == ["x"]


def test_annotate_chapters_enters_server_and_unloads(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    chapters = build_chapters(read_text(NOVEL), cfg.split.chapter_regex, cfg.split.max_chars)
    pipeline.annotate_chapters(chapters, cfg, out, True)
    assert fake_emotion.server_entered == 1
    assert fake_emotion.server_exited == 1
    assert fake_emotion.unloaded == 1


def test_annotate_progress_per_paragraph(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    chapters = build_chapters(read_text(NOVEL), cfg.split.chapter_regex, cfg.split.max_chars)
    total = sum(len(c.paragraphs) for c in chapters)
    calls = []
    pipeline.annotate_chapters(chapters, cfg, out, False, progress=lambda d, t, m: calls.append((d, t)))
    assert calls[-1] == (total, total) and len(calls) == total


# ---------- load_book ----------


def test_load_book_missing(tmp_path):
    with pytest.raises(ValueError, match="先に script"):
        pipeline.load_book(tmp_path / "nothing")


def test_load_book(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    pipeline.build_script(NOVEL, make_cfg(tmp_path), out, use_emotion=False)
    titles, rows = pipeline.load_book(out)
    assert set(titles) == {0, 1}
    keys = [(int(r["chapter"]), int(r["para"]), int(r["seg"])) for r in rows]
    assert keys == sorted(keys)
    assert pipeline.spoken_text(rows[0]) == rows[0]["text"]


# ---------- synthesize_book ----------


@pytest.fixture
def book(tmp_path, fake_emotion):
    out = tmp_path / "book"
    out.mkdir()
    cfg = make_cfg(tmp_path)
    pipeline.build_script(NOVEL, cfg, out, use_emotion=False)
    titles, rows = pipeline.load_book(out)
    return cfg, out, titles, rows


def test_synthesize_book_full(book):
    cfg, out, titles, rows = book
    synth = FakeSynth(out / "segments")
    res = pipeline.synthesize_book(rows, titles, cfg, out, synth)
    assert not res.stopped
    assert len(res.chapter_files) == 2
    assert res.chapter_files[0].parent.name == "chapters"
    assert res.chapter_files[0].name.startswith("01_") and res.chapter_files[0].suffix == ".wav"
    assert res.full_path == out / "book_full.wav" and res.full_path.exists()


def test_synthesize_book_subset_has_no_full(book):
    cfg, out, titles, rows = book
    res = pipeline.synthesize_book(rows, titles, cfg, out, FakeSynth(out / "segments"), chapters={1})
    assert len(res.chapter_files) == 1
    assert res.full_path is None


def test_synthesize_book_full_after_subsets(book):
    cfg, out, titles, rows = book
    synth = FakeSynth(out / "segments")
    pipeline.synthesize_book(rows, titles, cfg, out, synth, chapters={1})
    res = pipeline.synthesize_book(rows, titles, cfg, out, synth, chapters={2})
    assert res.full_path is not None and res.full_path.exists()


def test_synthesize_book_should_stop(book):
    cfg, out, titles, rows = book
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] >= 2

    synth = FakeSynth(out / "segments")
    res = pipeline.synthesize_book(rows, titles, cfg, out, synth, should_stop=should_stop)
    assert res.stopped and res.full_path is None
    assert len(synth.texts) == 1
    assert not (out / "chapters").exists() or not list((out / "chapters").glob("*.wav"))


def test_synthesize_book_force_and_progress(book):
    cfg, out, titles, rows = book
    synth = FakeSynth(out / "segments")
    prog = []
    pipeline.synthesize_book(rows, titles, cfg, out, synth, force=True, progress=lambda i, n, m: prog.append((i, n, m)))
    assert synth.forces and all(synth.forces)
    assert prog[-1][0] == prog[-1][1] == len(rows)
    assert "章" in prog[0][2]


# ---------- cli ----------


def test_cli_script_no_emotion(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert cli.main(["script", str(NOVEL), "-c", str(cfg), "--no-emotion"]) == 0
    out = tmp_path / "out" / "sample_novel"
    assert (out / "script.tsv").exists() and (out / "chapters.json").exists()
    assert "script.tsv を出力しました" in capsys.readouterr().out


def test_cli_script_no_chapters(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert cli.main(["script", str(NOVEL), "-c", str(cfg), "--no-emotion", "--chapters", "9"]) == 1
    assert "対象の章がありません" in capsys.readouterr().out


def test_cli_synth_without_torch(tmp_path, monkeypatch, capsys):
    cfg = write_cfg(tmp_path)
    cli.main(["script", str(NOVEL), "-c", str(cfg), "--no-emotion"])

    def boom(*a, **k):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(cli, "_make_synthesizer", boom)
    assert cli.main(["synth", str(NOVEL), "-c", str(cfg)]) == 2
    out = capsys.readouterr().out
    assert "torch" in out and "uv sync" in out and "No module named" in out


def test_cli_synth_without_script(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert cli.main(["synth", str(NOVEL), "-c", str(cfg)]) == 1
    assert "先に script" in capsys.readouterr().out


def test_cli_synth_without_voice(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"output_dir: {tmp_path / 'out'}\n", encoding="utf-8")
    cli.main(["script", str(NOVEL), "-c", str(cfg), "--no-emotion"])
    assert cli.main(["synth", str(NOVEL), "-c", str(cfg)]) == 1
    assert "voice" in capsys.readouterr().out


def test_cli_run_no_emotion(tmp_path, monkeypatch):
    cfg = write_cfg(tmp_path)
    monkeypatch.setattr(
        cli, "_make_synthesizer",
        lambda cfg, out_dir, voice, checkpoint: FakeSynth(out_dir / "segments"),
    )
    assert cli.main(["run", str(NOVEL), "-c", str(cfg), "--no-emotion"]) == 0
    out = tmp_path / "out" / "sample_novel"
    assert len(list((out / "chapters").glob("*.wav"))) == 2
    assert (out / "sample_novel_full.wav").exists()
