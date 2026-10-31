from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import pipeline
from irodori_speaker.config import Config
from irodori_speaker.synth import row_seed
from irodori_speaker.text_splitter import build_chapters

NOVEL = Path(__file__).parent / "sample_novel.txt"


class SeedSynth:
    latent_key = "L"

    def __init__(self, segments_dir: Path) -> None:
        self.segments_dir = segments_dir
        self.calls: list[tuple[str, int | None]] = []
        self.last_warnings: list[str] = []

    def synthesize(self, text, force=False, seed=None):
        self.calls.append((text, seed))
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        path = self.segments_dir / f"{hashlib.sha1(text.encode()).hexdigest()}.wav"
        sf.write(path, np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_16")
        return path


@pytest.fixture
def book(tmp_path):
    out = tmp_path / "book"
    cfg = Config(output_dir=str(tmp_path / "out"))
    pipeline.build_script(NOVEL, cfg, out, use_emotion=False)
    titles, rows = pipeline.load_book(out)
    return cfg, out, titles, rows


def _set_seed(out: Path, key: tuple[int, int, int], seed: str) -> None:
    rows = pipeline.load_script(out / "script.tsv")
    rows[key]["seed"] = seed
    pipeline.write_script(out / "script.tsv", [r for _, r in sorted(rows.items())])


def test_TSV_FIELDS_の末尾は_seed():
    assert pipeline.TSV_FIELDS[-1] == "seed"


def test_load_script_は_seed_列の無い古いファイルも読める(tmp_path):
    p = tmp_path / "script.tsv"
    old = [f for f in pipeline.TSV_FIELDS if f != "seed"]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=old, delimiter="\t")
        w.writeheader()
        w.writerow({"chapter": 0, "para": 0, "seg": 0, "kind": "narration", "labels": "n", "text": "あ", "source": "あ"})
    got = pipeline.load_script(p)[(0, 0, 0)]
    assert got["seed"] == ""


def test_row_seed():
    assert row_seed({"seed": ""}) is None
    assert row_seed({}) is None
    assert row_seed({"seed": "5"}) == 5
    assert row_seed({"seed": " 12 "}) == 12


def test_新しい文の_seed_は空(book):
    _cfg, _out, _titles, rows = book
    assert all(r["seed"] == "" for r in rows)


def test_手直しした_seed_は_script_再実行で保たれる(book):
    cfg, out, _titles, rows = book
    key = (int(rows[0]["chapter"]), int(rows[0]["para"]), int(rows[0]["seg"]))
    _set_seed(out, key, "99")
    again = pipeline.build_script(NOVEL, cfg, out, use_emotion=False)
    assert pipeline.load_script(out / "script.tsv")[key]["seed"] == "99"
    assert [r for r in again if (int(r["chapter"]), int(r["para"]), int(r["seg"])) == key][0]["seed"] == "99"


def test_段落の一部が変わっても一致する行の_seed_は保たれる(book):
    cfg, out, _titles, rows = book
    para_rows = [r for r in rows if (r["chapter"], r["para"]) == (rows[0]["chapter"], rows[0]["para"])]
    assert len(para_rows) >= 2, "サンプルの最初の段落が 2 文以上ある前提"
    first = (int(para_rows[0]["chapter"]), int(para_rows[0]["para"]), int(para_rows[0]["seg"]))
    last = (int(para_rows[-1]["chapter"]), int(para_rows[-1]["para"]), int(para_rows[-1]["seg"]))
    _set_seed(out, first, "11")
    _set_seed(out, last, "22")
    # 最後の文の source を変えて「段落の一部が変わった」状態にする（source は原文との一致判定）
    saved = pipeline.load_script(out / "script.tsv")
    saved[last]["source"] = "別の本文"
    pipeline.write_script(out / "script.tsv", [r for _, r in sorted(saved.items())])
    pipeline.build_script(NOVEL, cfg, out, use_emotion=False)
    got = pipeline.load_script(out / "script.tsv")
    assert got[first]["seed"] == "11"
    assert got[last]["seed"] == ""  # source が変わった文は作り直し（seed も空）


def test_synthesize_book_は行の_seed_を渡す(book):
    cfg, out, titles, rows = book
    rows = [dict(r, seed="7") if i == 0 else r for i, r in enumerate(rows)]
    synth = SeedSynth(out / "segments")
    pipeline.synthesize_book(rows, titles, cfg, out, synth)
    assert synth.calls[0][1] == 7
    assert all(seed is None for _, seed in synth.calls[1:])


def test_署名は_seed_があるときだけ変わる(book):
    cfg, out, _titles, rows = book
    ch = [r for r in rows if int(r["chapter"]) == 0]
    synth = SeedSynth(out / "segments")
    base = pipeline.chapter_signature(ch, cfg, synth)
    # seed 列が空の行は、seed 列の無かった頃と同じ署名（既存の .sig を無効にしない）
    without = [{k: v for k, v in r.items() if k != "seed"} for r in ch]
    assert pipeline.chapter_signature(without, cfg, synth) == base
    seeded = [dict(ch[0], seed="3"), *ch[1:]]
    assert pipeline.chapter_signature(seeded, cfg, synth) != base
    other = [dict(ch[0], seed="4"), *ch[1:]]
    assert pipeline.chapter_signature(other, cfg, synth) != pipeline.chapter_signature(seeded, cfg, synth)
