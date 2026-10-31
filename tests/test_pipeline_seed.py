from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import check, cli, pipeline
from irodori_speaker.config import Config
from irodori_speaker.synth import Synthesizer, row_seed

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
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["kind"] not in ("title", "break"):
            groups.setdefault((r["chapter"], r["para"]), []).append(r)
    para_rows = next((g for g in groups.values() if len(g) >= 2), None)
    assert para_rows is not None, "サンプルに 2 文以上の段落がある前提"
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


# ---------- check_book ----------

SR = 16000


def _wav(path: Path, seconds: float) -> None:
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)


def _check_cfg(tmp_path: Path) -> Config:
    return Config(output_dir=str(tmp_path / "out"), voice="none", device="cpu")


def _make_check_book(tmp_path, specs, cfg=None):
    """specs: (chapter, para, seg, kind, text, seconds|None, seed)。seconds が None なら WAV を作らない。"""
    cfg = cfg or _check_cfg(tmp_path)
    out = Path(cfg.output_dir) / "novel"
    out.mkdir(parents=True)
    chapters = sorted({s[0] for s in specs})
    (out / "chapters.json").write_text(
        json.dumps({"chapters": [{"idx": c, "title": f"t{c}"} for c in chapters]}), encoding="utf-8"
    )
    rows = [
        {"chapter": c, "para": p, "seg": s, "kind": kind, "labels": "", "text": text, "source": text, "seed": seed}
        for c, p, s, kind, text, _sec, seed in specs
    ]
    pipeline.write_script(out / "script.tsv", rows)
    synth = Synthesizer(cfg, out / "segments", [], Path(cfg.output_dir) / ".voices")
    keys = []
    for row, spec in zip(rows, specs):
        key = check.segment_key(synth, row)
        keys.append(key)
        if spec[5] is not None and spec[3] != "break":
            _wav(out / "segments" / f"{key}.wav", spec[5])
    return cfg, out, keys


TEXT = "あいうえおかきくけこ"  # 10 文字


def _text(i: int) -> str:
    """同じ長さで中身の違う文（同じ本文だと同じ鍵になって WAV を共有してしまう）。"""
    return "あいうえおかきくけ" + "こさしすせそたちつてと"[i]


def _normal_specs(n=8, long_at=None, missing_at=None):
    specs = []
    for i in range(n):
        sec = 1.0 + 0.01 * i
        if i == long_at:
            sec = 6.0
        specs.append((0, 0, i, "narration", _text(i), None if i == missing_at else sec, ""))
    return specs


def test_check_book_は全行を調べて_check_tsv_を書く(tmp_path):
    cfg, out, keys = _make_check_book(tmp_path, _normal_specs(long_at=3, missing_at=5))
    res = pipeline.check_book(cfg, out)
    assert res.path == out / "check.tsv" and res.path.exists()
    assert res.total == 8
    by_seg = {r.seg: r for r in res.rows}
    assert "long" in by_seg[3].flags
    assert by_seg[5].flags == ["missing"] and by_seg[5].stats is None
    assert by_seg[0].flags == []
    assert res.flagged == 2
    assert by_seg[0].key == keys[0]


def test_check_book_は_warnings_log_のキーに_warned_を付ける(tmp_path):
    cfg, out, keys = _make_check_book(tmp_path, _normal_specs())
    (out / "warnings.log").write_text(f"{keys[2]}\t{_text(2)}\tmax_seconds で切り詰め\n", encoding="utf-8")
    res = pipeline.check_book(cfg, out)
    assert [r.seg for r in res.rows if "warned" in r.flags] == [2]


def test_check_book_は区切り行を飛ばす(tmp_path):
    specs = [(0, 0, 0, "break", "", None, ""), *_normal_specs(3)[0:0]]
    specs += [(0, 1, i, "narration", TEXT, 1.0, "") for i in range(3)]
    cfg, out, _ = _make_check_book(tmp_path, specs)
    res = pipeline.check_book(cfg, out)
    assert res.total == 3 and all(r.para == 1 for r in res.rows)


def test_check_book_は行の_seed_を鍵に使う(tmp_path):
    specs = [(0, 0, 0, "narration", TEXT, 1.0, "9"), (0, 0, 1, "narration", TEXT, 1.0, "")]
    cfg, out, keys = _make_check_book(tmp_path, specs)
    res = pipeline.check_book(cfg, out)
    assert res.rows[0].key == keys[0] != res.rows[1].key
    assert all(r.stats is not None for r in res.rows)


def test_check_book_は章で絞る(tmp_path):
    specs = [(0, 0, 0, "narration", TEXT, 1.0, ""), (1, 0, 0, "narration", TEXT, 1.0, "")]
    cfg, out, _ = _make_check_book(tmp_path, specs)
    res = pipeline.check_book(cfg, out, chapters={2})
    assert [r.chapter for r in res.rows] == [1]


def test_check_book_は章_WAV_の_times_から_offset_を出す(tmp_path):
    cfg, out, _ = _make_check_book(tmp_path, [(0, 0, 0, "narration", TEXT, 1.0, "")])
    wav = pipeline._chapter_path(out, 0, {0: "t0"})
    wav.parent.mkdir(parents=True)
    (wav.with_suffix(".times.json")).write_text(
        json.dumps({"sr": 100, "frames": 1000, "paragraphs": [[0, 250, 900]]}), encoding="utf-8"
    )
    res = pipeline.check_book(cfg, out)
    assert res.rows[0].offset == pytest.approx(2.5)


def test_check_book_は_asr_で_cer_と_transcript_を付ける(tmp_path):
    cfg, out, _ = _make_check_book(tmp_path, _normal_specs(3))
    seen = []

    def fake(paths):
        seen.append(len(paths))
        return [_text(0), "まったくちがうぶんしょう", _text(2)]

    res = pipeline.check_book(cfg, out, asr=True, transcriber=fake)
    assert seen == [3]
    assert [r.transcript for r in res.rows][0] == _text(0)
    assert res.rows[0].cer == 0.0 and res.rows[1].cer > 0.3
    assert "cer" in res.rows[1].flags
    header = res.path.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert header[-2:] == ["cer", "transcript"]


def test_check_book_は_asr_に存在する_WAV_だけ渡す(tmp_path):
    cfg, out, _ = _make_check_book(tmp_path, _normal_specs(3, missing_at=1))
    got = []
    res = pipeline.check_book(cfg, out, asr=True, transcriber=lambda paths: got.append(paths) or [_text(0), _text(2)])
    assert len(got[0]) == 2
    assert res.rows[1].cer is None and res.rows[1].flags == ["missing"]


def test_check_book_は_voice_が無ければ_ValueError(tmp_path):
    cfg, out, _ = _make_check_book(tmp_path, _normal_specs(2))
    with pytest.raises(ValueError, match="voice"):
        pipeline.check_book(replace(cfg, voice=None), out)


def test_check_book_は台本が無ければ_ValueError(tmp_path):
    with pytest.raises(ValueError):
        pipeline.check_book(_check_cfg(tmp_path), tmp_path / "none")


# ---------- cli check ----------


def _cli_novel(tmp_path):
    novel = tmp_path / "novel.txt"
    novel.write_text("x", encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"output_dir: {tmp_path / 'out'}\ndevice: cpu\n", encoding="utf-8")
    return novel, cfg_path


def test_cli_check_は_script_が無ければ_1(tmp_path, capsys):
    novel, cfg_path = _cli_novel(tmp_path)
    assert cli.main(["check", str(novel), "-c", str(cfg_path), "--voice", "none"]) == 1
    assert "script" in capsys.readouterr().out


def test_cli_check_は要約を表示して印があっても_0(tmp_path, capsys):
    novel, cfg_path = _cli_novel(tmp_path)
    _make_check_book(tmp_path, _normal_specs(8, long_at=3), cfg=_check_cfg(tmp_path))
    assert cli.main(["check", str(novel), "-c", str(cfg_path), "--voice", "none"]) == 0
    assert "8 文中 1 件に印" in capsys.readouterr().out


def test_cli_check_は声が無ければ_1(tmp_path, capsys):
    novel, cfg_path = _cli_novel(tmp_path)
    _make_check_book(tmp_path, _normal_specs(2), cfg=_check_cfg(tmp_path))
    assert cli.main(["check", str(novel), "-c", str(cfg_path)]) == 1
    assert "voice" in capsys.readouterr().out


def test_cli_check_の_checkpoint_は鍵に反映される(tmp_path, capsys):
    novel, cfg_path = _cli_novel(tmp_path)
    cfg = replace(_check_cfg(tmp_path), checkpoint="my/ckpt")
    _make_check_book(tmp_path, _normal_specs(4), cfg=cfg)
    args = ["check", str(novel), "-c", str(cfg_path), "--voice", "none"]
    cli.main(args)
    assert "4 文中 4 件に印" in capsys.readouterr().out  # 既定の checkpoint では鍵が合わず全部 missing
    cli.main([*args, "--checkpoint", "my/ckpt"])
    assert "4 文中 0 件に印" in capsys.readouterr().out


def test_cli_check_は_asr_で_transformers_が無ければ_2(tmp_path, monkeypatch, capsys):
    novel, cfg_path = _cli_novel(tmp_path)
    _make_check_book(tmp_path, _normal_specs(2), cfg=_check_cfg(tmp_path))

    def boom(_cfg):
        raise ImportError("No module named 'transformers'")

    monkeypatch.setattr(check, "transcribe_with", boom)
    assert cli.main(["check", str(novel), "-c", str(cfg_path), "--voice", "none", "--asr"]) == 2
    assert "transformers" in capsys.readouterr().out


# ---------- 読み辞書・LoRA とキーを揃える ----


def _fake_runtime_synth(monkeypatch, cfg, out, readings=None):
    """runtime を偽にした本物の Synthesizer（segment_key は本物）。"""
    import types

    from irodori_speaker import synth as synth_mod

    monkeypatch.setattr(synth_mod, "_sampling_request", lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(
        synth_mod, "_save_wav",
        lambda path, audio, sr: sf.write(path, np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_16"),
    )
    runtime = types.SimpleNamespace(
        synthesize=lambda req: types.SimpleNamespace(audio="A", sample_rate=48000, messages=[])
    )
    monkeypatch.setattr(Synthesizer, "runtime", property(lambda self: runtime))
    return Synthesizer(cfg, out / "segments", [], Path(cfg.output_dir) / ".voices", readings=readings)


def test_check_は読み辞書のある本で_synthesize_book_の_WAV_と一致する(tmp_path, monkeypatch):
    specs = [(0, 0, 0, "narration", "東京へ行く", None, ""), (0, 0, 1, "narration", "東京から来た", None, "5")]
    cfg, out, _ = _make_check_book(tmp_path, specs)
    (out / "readings.yaml").write_text("東京: とうきょう\n", encoding="utf-8")
    from irodori_speaker.readings import load_readings

    synth = _fake_runtime_synth(monkeypatch, cfg, out, readings=load_readings(cfg, out))
    titles, rows = pipeline.load_book(out)
    pipeline.synthesize_book(rows, titles, cfg, out, synth)
    res = pipeline.check_book(cfg, out)
    assert all(r.stats is not None for r in res.rows), [r.flags for r in res.rows]
    assert not any("missing" in r.flags for r in res.rows)


def test_check_は_LoRA_のある本で_synthesize_book_の_WAV_と一致する(tmp_path, monkeypatch):
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_bytes(b"{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"W")
    cfg = replace(_check_cfg(tmp_path), lora_adapter=str(adapter))
    cfg, out, _ = _make_check_book(tmp_path, [(0, 0, 0, "narration", "あいう", None, "")], cfg=cfg)
    synth = _fake_runtime_synth(monkeypatch, cfg, out)
    titles, rows = pipeline.load_book(out)
    pipeline.synthesize_book(rows, titles, cfg, out, synth)
    assert pipeline.check_book(cfg, out).rows[0].stats is not None
    # LoRA を外した cfg では鍵が違う（= LoRA がキーに入っている）
    assert pipeline.check_book(replace(cfg, lora_adapter=None), out).rows[0].flags == ["missing"]


def test_cli_check_は_lora_adapter_を_cfg_に反映する(tmp_path, monkeypatch, capsys):
    novel, cfg_path = _cli_novel(tmp_path)
    seen = []

    def fake(cfg, out_dir, **kw):
        seen.append(cfg.lora_adapter)
        return pipeline.CheckResult(out_dir / "check.tsv", [], 0, 0)

    monkeypatch.setattr(cli, "check_book", fake)
    cli.main(["check", str(novel), "-c", str(cfg_path), "--voice", "none", "--lora-adapter", "/x/ad"])
    cli.main(["check", str(novel), "-c", str(cfg_path), "--voice", "none", "--lora-adapter", "none"])
    assert seen == ["/x/ad", None]
