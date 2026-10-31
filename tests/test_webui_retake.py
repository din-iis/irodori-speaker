from __future__ import annotations

import json
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from irodori_speaker import check
from irodori_speaker.config import load_config
from irodori_speaker.pipeline import TSV_FIELDS, write_script
from irodori_speaker.synth import Synthesizer
from irodori_speaker.webui import (
    CHECK_VIEW_FIELDS,
    STATIC_COLUMNS,
    check_view_rows,
    dataframe_rows,
    next_seed,
    rows_from_dataframe,
)


def _row(seg: int, text: str = "あ", seed: str = "", kind: str = "narration") -> dict:
    return {
        "chapter": "0", "para": "0", "seg": str(seg), "kind": kind,
        "labels": "neutral:1", "text": text, "source": f"原文{seg}", "seed": seed,
    }


# ---- 純粋ヘルパー ----


def test_next_seed():
    assert next_seed("", 42) == 43
    assert next_seed(None, 42) == 43
    assert next_seed("5", 42) == 6
    assert next_seed("43.0", 42) == 44
    assert next_seed(7, None) == 8
    r = next_seed("", None)
    assert isinstance(r, int) and 0 <= r < 2**31


def test_next_seed_は整数でなければ_ValueError():
    with pytest.raises(ValueError):
        next_seed("abc", 42)


def test_rows_from_dataframe_は_seed_列を取り込む():
    original = [_row(0), _row(1)]
    df = [
        ["0", "0", "0", "narration", "neutral:1", "あ", "原文0", "43.0"],
        ["0", "0", "1", "narration", "neutral:1", "あ", "原文1", ""],
    ]
    got = rows_from_dataframe(df, original)
    assert [r["seed"] for r in got] == ["43", ""]


def test_rows_from_dataframe_は欠損の_seed_を空にする():
    got = rows_from_dataframe([["0", "0", "0", "", "", "あ", "", float("nan")]], [_row(0, seed="9")])
    assert got[0]["seed"] == ""


def test_rows_from_dataframe_は整数でない_seed_を拒否する():
    with pytest.raises(ValueError):
        rows_from_dataframe([["0", "0", "0", "", "", "あ", "", "abc"]], [_row(0)])


def test_dataframe_rows_は_seed_の無い行でも動く():
    old = {k: v for k, v in _row(0).items() if k != "seed"}
    assert dataframe_rows([old])[0][-1] == ""
    assert len(dataframe_rows([_row(0, seed="3")])[0]) == len(TSV_FIELDS)


def test_static_columns_は_text_と_seed_以外():
    assert [TSV_FIELDS[i] for i in STATIC_COLUMNS] == [f for f in TSV_FIELDS if f not in ("text", "seed")]


def _check_row(flags, seg=0) -> check.CheckRow:
    st = check.AudioStats(seconds=2.0, leading_silence=0, trailing_silence=0, tail_noise=0, clipping=0)
    return check.CheckRow(chapter=0, para=1, seg=seg, key="k", text="本文", chars=4, stats=st, z=3.456,
                          offset=None, flags=flags)


def test_check_view_rows_は印のある行だけを表の列にする():
    rows = check_view_rows([_check_row([]), _check_row(["long", "tail"], seg=2)])
    assert CHECK_VIEW_FIELDS == ["chapter", "para", "seg", "flags", "seconds", "chars", "z", "text"]
    assert rows == [["0", "1", "2", "long|tail", "2.00", "4", "3.46", "本文"]]


def test_check_view_rows_は_None_を空欄にする():
    row = _check_row(["missing"])
    row.stats, row.z = None, None
    assert check_view_rows([row]) == [["0", "1", "0", "missing", "", "4", "", "本文"]]


# ---- ハンドラ（gradio が要る） ----


def _handlers(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    from irodori_speaker import synth, webui

    monkeypatch.setattr(synth, "release_runtime", lambda: None)
    app = webui.build_app(replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out")))
    return {f.fn.__name__: f.fn for f in app.fns.values()}


class _FakeSynth:
    def __init__(self):
        self.calls = []
        self.last_warnings = []

    def synthesize(self, text, force=False, seed=None):
        self.calls.append((text, force, seed))
        return Path("/fake/seg.wav")


def _book(tmp_path, rows):
    cfg = replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out"))
    novel = tmp_path / "n.txt"
    novel.write_text("x", encoding="utf-8")
    out = Path(cfg.output_dir) / "n"
    out.mkdir(parents=True)
    (out / "chapters.json").write_text(json.dumps({"chapters": [{"idx": 0, "title": "t"}]}), encoding="utf-8")
    write_script(out / "script.tsv", rows)
    return cfg, novel, out


TTS = (None, "cpu", "fp32", "none")  # checkpoint, device, precision, voice（none なら resolve_voice を通らない）


def test_select_row_は空の選択なら何も変えない(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    for index in (None, []):
        got = fns["select_row"]([], [], None, "", *TTS, types.SimpleNamespace(index=index))
        assert got == ({"__type__": "update"},) * 3


def test_select_row_は区切り行には音声が無いと伝える(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    _cfg, novel, _out = _book(tmp_path, [_row(0, "", kind="break")])
    rows = [_row(0, "", kind="break")]
    idx, path, status = fns["select_row"](dataframe_rows(rows), rows, None, str(novel), *TTS,
                                          types.SimpleNamespace(index=[0, 0]))
    assert idx == 0 and path is None and "区切り" in status


def test_select_row_はキャッシュの音声のパスを返す(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "あ", seed="5"), _row(1, "い")]
    cfg, novel, out = _book(tmp_path, rows)
    synth = Synthesizer(replace(cfg, device="cpu", precision="fp32", voice="none"), out / "segments", [],
                        Path(cfg.output_dir) / ".voices")
    key = check.segment_key(synth, rows[0])
    sf.write(out / "segments" / f"{key}.wav", np.zeros(100, dtype=np.float32), 16000)
    evt = types.SimpleNamespace(index=[0, 5])
    idx, path, _status = fns["select_row"](dataframe_rows(rows), rows, None, str(novel), *TTS, evt)
    assert idx == 0 and path == str(out / "segments" / f"{key}.wav")
    # 音声が無い文は None
    idx, path, status = fns["select_row"](dataframe_rows(rows), rows, None, str(novel), *TTS,
                                          types.SimpleNamespace(index=[1, 0]))
    assert idx == 1 and path is None and "まだ" in status


def _retake_setup(monkeypatch, tmp_path, rows):
    from irodori_speaker import webui

    fns = _handlers(monkeypatch, tmp_path)
    fake = _FakeSynth()
    monkeypatch.setattr(webui.Engine, "is_loaded", lambda self, key: True)
    monkeypatch.setattr(webui.Engine, "load", lambda self, *a, **k: fake)
    cfg, novel, out = _book(tmp_path, rows)
    return fns, fake, novel


def test_retake_row_は_seed_を増やして表と_State_を更新する(monkeypatch, tmp_path):
    rows = [_row(0, "あ", seed=""), _row(1, "い", seed="10")]
    fns, fake, novel = _retake_setup(monkeypatch, tmp_path, rows)
    outs = list(fns["retake_row"](1, dataframe_rows(rows), rows, None, str(novel), *TTS))
    table, state, audio, status = outs[-1]
    assert fake.calls == [("い", False, 11)]
    assert state[1]["seed"] == "11" and state[0]["seed"] == ""
    assert table["value"][1][TSV_FIELDS.index("seed")] == "11"
    assert audio == "/fake/seg.wav" and "seed=11" in status


def test_retake_row_は_seed_が空なら_config_の_seed_に_1_を足す(monkeypatch, tmp_path):
    rows = [_row(0, "あ")]
    fns, fake, novel = _retake_setup(monkeypatch, tmp_path, rows)
    list(fns["retake_row"](0, dataframe_rows(rows), rows, None, str(novel), *TTS))
    assert fake.calls[0][2] == load_config("config.example.yaml").sampling.seed + 1


def test_retake_row_は表で直した_text_を使う(monkeypatch, tmp_path):
    rows = [_row(0, "あ")]
    fns, fake, novel = _retake_setup(monkeypatch, tmp_path, rows)
    df = dataframe_rows(rows)
    df[0][TSV_FIELDS.index("text")] = "直した"
    list(fns["retake_row"](0, df, rows, None, str(novel), *TTS))
    assert fake.calls[0][0] == "直した"


def test_retake_row_は選択なしと区切り行で_Error(monkeypatch, tmp_path):
    rows = [_row(0, "", kind="break")]
    fns, fake, novel = _retake_setup(monkeypatch, tmp_path, rows)
    import gradio as gr

    with pytest.raises(gr.Error):
        list(fns["retake_row"](None, dataframe_rows(rows), rows, None, str(novel), *TTS))
    with pytest.raises(gr.Error):
        list(fns["retake_row"](0, dataframe_rows(rows), rows, None, str(novel), *TTS))
    assert fake.calls == []


def test_run_check_は印の付いた行だけ表に出す(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "あ"), _row(1, "い")]
    cfg, novel, out = _book(tmp_path, rows)
    synth = Synthesizer(replace(cfg, device="cpu", precision="fp32", voice="none"), out / "segments", [],
                        Path(cfg.output_dir) / ".voices")
    t = np.arange(16000) / 16000
    sf.write(out / "segments" / f"{check.segment_key(synth, rows[0])}.wav",
             (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
    table, status = fns["run_check"](None, str(novel), "", *TTS, dataframe_rows(rows), rows)
    assert table == [["0", "0", "1", "missing", "", "1", "", "い"]]
    assert "2 文中 1 件に印" in status and "check.tsv" in status
