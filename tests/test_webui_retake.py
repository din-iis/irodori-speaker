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


# ---- 読み辞書・LoRA との統合、共有 Synthesizer を触らないこと ----


def _capture_engine(monkeypatch):
    from irodori_speaker import webui

    engines = []
    orig = webui.Engine.__init__

    def init(self, cfg):
        orig(self, cfg)
        engines.append(self)

    monkeypatch.setattr(webui.Engine, "__init__", init)
    return engines


def test_select_row_は_Engine_の共有_Synthesizer_を書き換えない(monkeypatch, tmp_path):
    from irodori_speaker.synth import segment_cache_key

    engines = _capture_engine(monkeypatch)
    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "東京", seed="5")]
    cfg, novel, out = _book(tmp_path, rows)
    (out / "readings.yaml").write_text("東京: とうきょう\n", encoding="utf-8")
    other_seg, other_warn = tmp_path / "other" / "segments", tmp_path / "other" / "warnings.log"
    shared = types.SimpleNamespace(
        segments_dir=other_seg, warnings_path=other_warn, readings={"x": "y"}, latent_key="L", lora_key="K",
    )
    engine = engines[-1]
    key = engine.key_for(*TTS)
    engine._synth, engine._key = shared, key
    expected = segment_cache_key(
        engine.cfg_for(key), "東京", "L", readings={"東京": "とうきょう"}, lora_key="K", seed=5
    )
    wav = out / "segments" / f"{expected}.wav"
    wav.parent.mkdir(exist_ok=True)
    sf.write(wav, np.zeros(100, dtype=np.float32), 16000)
    _idx, path, _status = fns["select_row"](dataframe_rows(rows), rows, None, str(novel), *TTS,
                                            types.SimpleNamespace(index=[0, 0]))
    assert path == str(wav)  # この本の読み辞書と、共有 Synthesizer の latent_key / lora_key で引く
    assert (shared.segments_dir, shared.warnings_path, shared.readings) == (other_seg, other_warn, {"x": "y"})
    assert engine._synth is shared and engine._key == key


def test_select_row_は共有が無ければ独立した_Synthesizer_で引き_Engine_に登録しない(monkeypatch, tmp_path):
    engines = _capture_engine(monkeypatch)
    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "あ")]
    _cfg, novel, _out = _book(tmp_path, rows)
    fns["select_row"](dataframe_rows(rows), rows, None, str(novel), *TTS, types.SimpleNamespace(index=[0, 0]))
    assert engines[-1]._synth is None and engines[-1]._key is None


def test_select_row_は_lora_を_tts_key_に使う(monkeypatch, tmp_path):
    engines = _capture_engine(monkeypatch)
    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "あ")]
    cfg, novel, out = _book(tmp_path, rows)
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_bytes(b"{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"W")
    from irodori_speaker.lora import lora_key
    from irodori_speaker.synth import segment_cache_key

    engine = engines[-1]
    key = engine.key_for(*TTS, str(adapter))
    expected = segment_cache_key(
        engine.cfg_for(key), "あ", engine._synth.latent_key if engine._synth else
        Synthesizer(engine.cfg_for(key), tmp_path / "s", [], tmp_path / "l").latent_key,
        lora_key=lora_key(adapter),
    )
    (out / "segments").mkdir(exist_ok=True)
    sf.write(out / "segments" / f"{expected}.wav", np.zeros(10, dtype=np.float32), 16000)
    _i, path, _s = fns["select_row"](dataframe_rows(rows), rows, None, str(novel), *TTS,
                                     types.SimpleNamespace(index=[0, 0]), lora_adapter=str(adapter))
    assert path and path.endswith(f"{expected}.wav")


def test_retake_row_は読み辞書と_LoRA_のキーで_WAV_を書く(monkeypatch, tmp_path):
    from irodori_speaker import synth as synth_mod
    from irodori_speaker import webui

    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "東京", seed="")]
    cfg, novel, out = _book(tmp_path, rows)
    monkeypatch.setattr(synth_mod, "_sampling_request", lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(synth_mod, "_save_wav", lambda path, audio, sr: Path(path).write_bytes(b"RIFF"))
    runtime = types.SimpleNamespace(
        synthesize=lambda req: types.SimpleNamespace(audio="A", sample_rate=48000, messages=[])
    )
    monkeypatch.setattr(Synthesizer, "runtime", property(lambda self: runtime))
    real = Synthesizer(replace(cfg, device="cpu", precision="fp32", voice="none"), out / "segments", [],
                       tmp_path / "lat", readings={"東京": "とうきょう"})
    real.lora_key = "K"
    seen = []
    monkeypatch.setattr(webui.Engine, "is_loaded", lambda self, key: True)
    monkeypatch.setattr(webui.Engine, "load", lambda self, *a, **k: seen.append((a, k)) or real)
    outs = list(fns["retake_row"](0, dataframe_rows(rows), rows, None, str(novel), *TTS))
    table, state, audio, status = outs[-1]
    new_seed = int(state[0]["seed"])
    assert Path(audio).name == f"{real.segment_key('東京', seed=new_seed)}.wav" and Path(audio).exists()
    assert real.segment_key("東京", seed=new_seed) != Synthesizer(
        real.cfg, tmp_path / "x", [], tmp_path / "lat").segment_key("東京", seed=new_seed)
    (args, _kwargs), = seen  # out_dir を渡して、本ごとの読み辞書に差し替えさせる
    assert out in args or _kwargs.get("out_dir") == out


def test_run_check_は_lora_入りの_cfg_を_check_book_に渡す(monkeypatch, tmp_path):
    from irodori_speaker import pipeline, webui

    fns = _handlers(monkeypatch, tmp_path)
    rows = [_row(0, "あ")]
    _cfg, novel, out = _book(tmp_path, rows)
    seen = []

    def fake(cfg, out_dir, **kw):
        seen.append(cfg.lora_adapter)
        return pipeline.CheckResult(out_dir / "check.tsv", [], 0, 0)

    monkeypatch.setattr(webui, "check_book", fake)
    fns["run_check"](None, str(novel), "", *TTS, dataframe_rows(rows), rows, lora_adapter="/x/ad")
    assert seen == ["/x/ad"]


# ---- 小説の切り替え・台本の作り直しで行の選択を消す ----

NO_UPDATE = {"__type__": "update"}


def test_小説を切り替えると選択と音声が消える(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    cfg, novel_a, _out = _book(tmp_path, [_row(0, "あ")])
    other = tmp_path / "b.txt"
    other.write_text("x", encoding="utf-8")
    table, rows, status, loaded, sel, audio = fns["show_script_clearing"](None, str(novel_a), None)
    assert (sel, audio) == (None, None) and loaded == str(novel_a)
    # 別の小説に切り替えたとき（script.tsv が無くても）選択を消す
    *_, sel, audio = fns["show_script_clearing"](None, str(other), loaded)
    assert (sel, audio) == (None, None)


def test_同じ小説の再表示では選択を保つ(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    cfg, novel, _out = _book(tmp_path, [_row(0, "あ")])
    *_, loaded, _s, _a = fns["show_script_clearing"](None, str(novel), None)
    *_, sel, audio = fns["show_script_clearing"](None, str(novel), loaded)
    assert (sel, audio) == (NO_UPDATE, NO_UPDATE)


def test_台本を作り直すと選択と音声が消える(monkeypatch, tmp_path):
    fns = _handlers(monkeypatch, tmp_path)
    novel = tmp_path / "n.txt"
    novel.write_text("第一章\n本文です。\n", encoding="utf-8")
    outs = list(fns["run_script"](None, str(novel), "", True, False, False, None, "cpu", "fp32", "none",
                                  "ollama", "http://x", "m", False, 1, [], []))
    assert outs[-1][-2:] == (None, None)
    assert all(o[-2:] == (NO_UPDATE, NO_UPDATE) for o in outs if not isinstance(o[2], list))


def test_取り直しは選択を消さない(monkeypatch, tmp_path):
    rows = [_row(0, "あ")]
    fns, fake, novel = _retake_setup(monkeypatch, tmp_path, rows)
    outs = list(fns["retake_row"](0, dataframe_rows(rows), rows, None, str(novel), *TTS))
    assert all(len(o) == 4 for o in outs)  # selected_idx を出力に持たない = 選択は State に残る
