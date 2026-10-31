"""BGM の音量の物差し（K 特性で重み付けした音量）と、CLI / WebUI からの上書き。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from irodori_speaker import bgm_mix
from irodori_speaker.bgm_mix import Layout, Track, loudness_db, mix_signature, render
from irodori_speaker.bgm_types import Scene
from irodori_speaker.config import BgmConfig, load_config

SR = 24000


def _sine(freq: float, seconds: float, rms_db: float, sr: int = SR) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) * 10 ** (rms_db / 20) * np.sqrt(2)).astype("float32")


def _band_noise(lo: float, hi: float, seconds: float, rms_db: float, sr: int = SR) -> np.ndarray:
    """lo〜hi Hz の帯域雑音（周波数領域で作る）。RMS を rms_db に揃える。"""
    n = round(seconds * sr)
    spec = np.fft.rfft(np.random.default_rng(1).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    x *= 10 ** (rms_db / 20) / np.sqrt(np.mean(x**2))
    return x.astype("float32")


def test_loudness_db_997hz_sine_is_its_rms_level():
    # K 特性の利得は 997 Hz で +0.691 dB、LUFS のオフセットが -0.691 なので RMS と一致する
    assert loudness_db(_sine(997, 3, -20.0), SR) == pytest.approx(-20.0, abs=0.1)


def test_loudness_db_high_frequency_reads_louder_than_low():
    low = loudness_db(_sine(100, 3, -20.0), SR)
    high = loudness_db(_sine(4000, 3, -20.0), SR)
    assert high - low > 3.0  # 高域シェルフ（約 +4 dB）と低域の減衰


def test_loudness_db_silence_is_minus_inf():
    assert loudness_db(np.zeros(SR, dtype="float32"), SR) == -np.inf
    assert loudness_db(np.zeros(0, dtype="float32"), SR) == -np.inf


def test_loudness_db_gate_ignores_long_silence():
    voice = _sine(997, 2, -20.0)
    padded = np.concatenate([voice, np.zeros(20 * SR, dtype="float32"), voice])
    assert loudness_db(padded, SR) == pytest.approx(-20.0, abs=0.3)


def test_loudness_db_same_signal_at_24k_and_48k():
    a = _sine(1000, 3, -20.0, 24000) + _sine(3000, 3, -26.0, 24000)
    b = _sine(1000, 3, -20.0, 48000) + _sine(3000, 3, -26.0, 48000)
    assert loudness_db(a, 24000) == pytest.approx(loudness_db(b, 48000), abs=0.3)


def test_loudness_db_stereo_sums_channels():
    mono = _sine(997, 3, -20.0)
    stereo = np.stack([mono, mono], axis=1)
    assert loudness_db(stereo, SR) - loudness_db(mono, SR) == pytest.approx(10 * np.log10(2), abs=0.05)


def test_loudness_db_short_signal_is_one_block():
    assert loudness_db(_sine(997, 0.1, -20.0), SR) == pytest.approx(-20.0, abs=0.5)


def _render_ambient(material: np.ndarray, voice_db: float = -20.0, bgm: BgmConfig | None = None):
    n = 10 * SR
    narration = _sine(997, 10, voice_db)
    layout = Layout(total=n, voice=[], ambient=[Track("a", 0, n, 0, 0)], music=[])
    # 声を重ねずに環境音だけを取り出す
    return render(layout, narration, {"a": material}, SR, bgm or BgmConfig())


def test_render_high_frequency_noise_is_mixed_quieter_than_low():
    low = _band_noise(50, 400, 5, -15.0)
    high = _band_noise(3000, 8000, 5, -15.0)
    low_rms = np.sqrt(np.mean(_render_ambient(np.stack([low, low], 1))[SR:] ** 2))
    high_rms = np.sqrt(np.mean(_render_ambient(np.stack([high, high], 1))[SR:] ** 2))
    assert high_rms < low_rms * 10 ** (-3.0 / 20)


def test_render_ambient_loudness_matches_voice_plus_ambient_db():
    bgm = BgmConfig(ambient_db=-20.0)
    for lo, hi in ((50, 400), (3000, 8000)):
        x = _band_noise(lo, hi, 10, -15.0)  # ループのつなぎ目で音量が揺れないよう、長さを揃える
        out = _render_ambient(np.stack([x, x], 1), voice_db=-20.0, bgm=bgm)
        # 声は dual mono（両チャンネル同じ）として測る = +3.01 dB
        voice = loudness_db(_sine(997, 10, -20.0), SR) + 10 * np.log10(2)
        assert loudness_db(out[SR:], SR) == pytest.approx(voice + bgm.ambient_db, abs=0.5)


def test_mix_rule_is_2_and_changes_signature(monkeypatch):
    assert bgm_mix.MIX_RULE == 2
    scene = Scene(
        chapter=0, index=0, start_para=0, end_para=0, description="d",
        ambient_query=None, mood="calm", music_query="p",
    )
    sig = mix_signature("a", [scene], {}, BgmConfig())
    monkeypatch.setattr(bgm_mix, "MIX_RULE", 1)
    assert mix_signature("a", [scene], {}, BgmConfig()) != sig


# -- CLI --


def _cli_run(monkeypatch, tmp_path, extra):
    from irodori_speaker import cli
    from irodori_speaker.pipeline import BgmResult

    seen = {}

    def fake(cfg, out_dir, **kw):
        seen["cfg"] = cfg
        return BgmResult([], None, 0)

    monkeypatch.setattr(cli, "build_bgm", fake)
    novel = tmp_path / "n.txt"
    novel.write_text("第一章\n本文です。\n", encoding="utf-8")
    pytest.importorskip("requests")
    rc = cli.main(["bgm", str(novel), "-c", "config.example.yaml", *extra])
    assert rc == 0
    return seen["cfg"]


def test_cli_bgm_overrides_ambient_and_music_db(monkeypatch, tmp_path):
    cfg = _cli_run(monkeypatch, tmp_path, ["--ambient-db", "-27", "--music-db", "-6.5"])
    assert cfg.bgm.ambient_db == -27.0
    assert cfg.bgm.music_db == -6.5


def test_cli_bgm_keeps_config_values_without_flags(monkeypatch, tmp_path):
    base = load_config("config.example.yaml").bgm
    cfg = _cli_run(monkeypatch, tmp_path, [])
    assert cfg.bgm.ambient_db == base.ambient_db
    assert cfg.bgm.music_db == base.music_db


# -- WebUI --


def _webui(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    pytest.importorskip("requests")
    from irodori_speaker import synth, webui
    from irodori_speaker.pipeline import BgmResult

    monkeypatch.setattr(synth, "release_runtime", lambda: None)
    cfg = replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out"))
    app = webui.build_app(cfg)
    fns = {f.fn.__name__: f.fn for f in app.fns.values()}
    novel = tmp_path / "n.txt"
    novel.write_text("第一章\n本文です。\n", encoding="utf-8")
    seen = {}

    def fake(c, out_dir, **kw):
        seen["cfg"] = c
        return BgmResult([], None, 0)

    monkeypatch.setattr(webui, "build_bgm", fake)
    return cfg, fns, novel, seen


def _add(fns, novel, *levels):
    return list(fns["add_bgm"](None, str(novel), "", "ollama", "http://x", "m", True, 2, *levels))


def test_webui_sliders_override_cfg_and_show_in_status(monkeypatch, tmp_path):
    cfg, fns, novel, seen = _webui(monkeypatch, tmp_path)
    outs = _add(fns, novel, -25, -6)
    assert seen["cfg"].bgm.ambient_db == -25
    assert seen["cfg"].bgm.music_db == -6
    assert "環境音 -25 dB / 音楽 -6 dB（声に対して）" in outs[-1][0]


def test_webui_none_levels_keep_config(monkeypatch, tmp_path):
    cfg, fns, novel, seen = _webui(monkeypatch, tmp_path)
    outs = _add(fns, novel)
    assert seen["cfg"].bgm == cfg.bgm
    assert f"環境音 {cfg.bgm.ambient_db:g} dB / 音楽 {cfg.bgm.music_db:g} dB（声に対して）" in outs[-1][0]


def test_webui_sliders_start_at_config_values(monkeypatch, tmp_path):
    gr = pytest.importorskip("gradio")
    cfg, fns, novel, seen = _webui(monkeypatch, tmp_path)
    from irodori_speaker import webui

    app = webui.build_app(cfg)
    sliders = [b for b in app.blocks.values() if isinstance(b, gr.Slider) and "声に対して" in (b.label or "")]
    assert sorted(s.value for s in sliders) == sorted([cfg.bgm.ambient_db, cfg.bgm.music_db])
