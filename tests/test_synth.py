import sys
import types
from pathlib import Path

import pytest

from irodori_speaker import synth
from irodori_speaker.config import Config
from irodori_speaker.synth import (
    Synthesizer,
    cache_key,
    check_vram,
    collect_warnings,
    latent_key,
    latent_key_for,
)


def test_latent_key_deterministic_and_sensitive():
    base = latent_key("v", "codec", -16.0, None)
    assert base == latent_key("v", "codec", -16.0, None)
    assert base != latent_key("w", "codec", -16.0, None)
    assert base != latent_key("v", "other", -16.0, None)
    assert base != latent_key("v", "codec", None, None)
    assert base != latent_key("v", "codec", -16.0, 30.0)


def test_cache_key_deterministic_and_sensitive():
    cfg = Config()
    k = cache_key(cfg, "こんにちは", "L")
    assert k == cache_key(Config(), "こんにちは", "L")
    assert k != cache_key(cfg, "さようなら", "L")
    assert k != cache_key(cfg, "こんにちは", "M")
    other = Config()
    other.sampling.seed = 1
    assert k != cache_key(other, "こんにちは", "L")


def test_cache_key_ignores_device_and_precision():
    k = cache_key(Config(), "a", "L")
    assert k == cache_key(Config(device="cpu", precision="fp32"), "a", "L")


@pytest.mark.parametrize(
    "field,value",
    [("num_steps", 8), ("t_schedule_mode", "x"), ("sway_coeff", 0.5), ("cfg_scale_text", 1.0),
     ("cfg_scale_speaker", 1.0), ("duration_scale", 1.2), ("max_seconds", 10.0)],
)
def test_cache_key_sensitive_to_sampling(field, value):
    cfg = Config()
    setattr(cfg.sampling, field, value)
    assert cache_key(cfg, "a", "L") != cache_key(Config(), "a", "L")


def test_cache_key_sensitive_to_checkpoint():
    assert cache_key(Config(checkpoint="x"), "a", "L") != cache_key(Config(), "a", "L")


def test_collect_warnings_truncated():
    msgs = ["info: predicted duration frames=1500.0, scale=1.000, using_frames=1000 (30.000s)."]
    assert collect_warnings(msgs) == [
        "max_seconds で切り詰め: 予測 1500.0 フレーム × 1.000 > 1000（約 30.000s）"
    ]


def test_collect_warnings_truncated_fractional_scale():
    msgs = ["info: predicted duration frames=1200.0, scale=1.200, using_frames=1000 (30.000s)."]
    assert len(collect_warnings(msgs)) == 1


def test_collect_warnings_not_truncated():
    msgs = [
        "info: predicted duration frames=900.0, scale=1.000, using_frames=900 (18.000s).",
        "info: other",
    ]
    assert collect_warnings(msgs) == []


def test_collect_warnings_rounding_is_not_truncation():
    msgs = ["info: predicted duration frames=900.4, scale=1.000, using_frames=900 (18.000s)."]
    assert collect_warnings(msgs) == []


def test_collect_warnings_non_info_passthrough():
    assert collect_warnings(["text too long", "info: x"]) == ["text too long"]


def _fake_torch(monkeypatch, available=True, free_gb=8.0):
    calls = {"empty_cache": 0}
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=lambda device=None: (calls.__setitem__("device", device), (int(free_gb * 2**30), 8 * 2**30))[1],
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return calls


def test_check_vram_cpu_noop(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    check_vram(Config(device="cpu"))  # torch を import しようとすると失敗する


def test_check_vram_insufficient(monkeypatch):
    _fake_torch(monkeypatch, free_gb=1.5)
    with pytest.raises(RuntimeError, match=r"1\.5 GB.*3\.0 GB"):
        check_vram(Config(device="cuda", min_free_vram_gb=3.0))


def test_check_vram_sufficient(monkeypatch):
    _fake_torch(monkeypatch, free_gb=6.0)
    check_vram(Config(device="cuda", min_free_vram_gb=3.0))


def test_check_vram_cuda_unavailable(monkeypatch):
    _fake_torch(monkeypatch, available=False, free_gb=0.1)
    check_vram(Config(device="cuda", min_free_vram_gb=3.0))


def test_check_vram_disabled(monkeypatch):
    _fake_torch(monkeypatch, free_gb=0.1)
    check_vram(Config(device="cuda", min_free_vram_gb=0))


class FakeRuntime:
    def __init__(self, messages=()):
        self.calls = []
        self.messages = list(messages)

    def synthesize(self, req):
        self.calls.append(req)
        return types.SimpleNamespace(audio="AUDIO", sample_rate=48000, messages=self.messages)


@pytest.fixture
def make_synth(tmp_path, monkeypatch):
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"abc")

    def fake_request(**kw):
        return types.SimpleNamespace(**kw)

    def fake_save(path, audio, sr):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr(synth, "_sampling_request", fake_request)
    monkeypatch.setattr(synth, "_save_wav", fake_save)

    def make(cfg=None, runtime=None, warnings=True):
        cfg = cfg or Config(device="cpu")
        s = Synthesizer(cfg, tmp_path / "seg", [voice], tmp_path / "lat",
                        tmp_path / "warn.tsv" if warnings else None)
        rt = runtime or FakeRuntime()
        monkeypatch.setattr(Synthesizer, "runtime", property(lambda self: rt))
        monkeypatch.setattr(Synthesizer, "latent_path", property(lambda self: tmp_path / "x.pt"))
        return s, rt

    return make


def test_synthesize_cache_hit_does_not_touch_runtime(make_synth, monkeypatch):
    s, _ = make_synth()
    path = s.segments_dir / f"{cache_key(s.cfg, 'hi', s.latent_key)}.wav"
    path.write_bytes(b"cached")

    def boom(self):
        raise AssertionError("runtime touched")

    monkeypatch.setattr(Synthesizer, "runtime", property(boom))
    monkeypatch.setattr(Synthesizer, "latent_path", property(boom))
    assert s.synthesize("hi") == path


def test_synthesize_force_calls_runtime(make_synth):
    s, rt = make_synth()
    path = s.segments_dir / f"{cache_key(s.cfg, 'hi', s.latent_key)}.wav"
    path.write_bytes(b"cached")
    assert s.synthesize("hi", force=True) == path
    assert len(rt.calls) == 1
    assert path.read_bytes() == b"RIFF"
    assert rt.calls[0].text == "hi" and rt.calls[0].seed == 42


def test_synthesize_miss_writes_wav_and_no_tmp_left(make_synth):
    s, rt = make_synth()
    path = s.synthesize("hi")
    assert path.read_bytes() == b"RIFF"
    assert [p.name for p in s.segments_dir.iterdir()] == [path.name]


def test_synthesize_appends_warnings(make_synth, tmp_path):
    msgs = ["bad thing", "info: predicted duration frames=100.0, scale=1.000, using_frames=50 (1.000s)."]
    s, _ = make_synth(runtime=FakeRuntime(msgs))
    path = s.synthesize("hi")
    assert len(s.last_warnings) == 2
    lines = (tmp_path / "warn.tsv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"{path.stem}\thi\tbad thing"
    assert len(lines) == 2


def test_synthesize_clears_last_warnings_when_clean(make_synth):
    s, _ = make_synth()
    s.last_warnings = ["old"]
    s.synthesize("hi")
    assert s.last_warnings == []


def test_empty_cache_interval(make_synth, monkeypatch):
    calls = _fake_torch(monkeypatch)
    cfg = Config(device="cuda", empty_cache_interval=2)
    s, _ = make_synth(cfg)
    for t in "abcde":
        s.synthesize(t)
    assert calls["empty_cache"] == 2


def test_empty_cache_not_called_on_cpu(make_synth, monkeypatch):
    calls = _fake_torch(monkeypatch)
    s, _ = make_synth(Config(device="cpu", empty_cache_interval=1))
    s.synthesize("a")
    assert calls["empty_cache"] == 0


def test_generate_once_returns_audio_without_caching(make_synth, monkeypatch):
    import numpy as np

    class Audio:
        def detach(self): return self
        def cpu(self): return self
        def float(self): return self
        def numpy(self): return np.ones((1, 4), dtype="float64")

    rt = FakeRuntime(["warn"])
    rt.synthesize = lambda req: (rt.calls.append(req), types.SimpleNamespace(
        audio=Audio(), sample_rate=24000, messages=["warn"]))[1]
    s, _ = make_synth(runtime=rt)
    sr, audio, warns = s.generate_once("hi", seed=None, num_steps=4)
    assert sr == 24000 and audio.shape == (4,) and audio.dtype == np.float32
    assert warns == ["warn"]
    assert rt.calls[0].seed is None and rt.calls[0].num_steps == 4
    assert rt.calls[0].duration_scale == 1.0
    assert not list(s.segments_dir.iterdir())


def test_latent_key_for_none_depends_on_checkpoint():
    assert latent_key_for(Config(checkpoint="a"), "v") != latent_key_for(Config(checkpoint="b"), "v")


def test_latent_key_for_explicit_ignores_checkpoint():
    a = latent_key_for(Config(checkpoint="a", max_ref_seconds=60), "v")
    assert a == latent_key_for(Config(checkpoint="b", max_ref_seconds=60), "v")


def test_synthesizer_latent_key_matches_helper(make_synth):
    s, _ = make_synth()
    assert s.latent_key == latent_key_for(s.cfg, synth._voice_key(s.voice_paths))


def test_check_vram_passes_device(monkeypatch):
    calls = _fake_torch(monkeypatch, free_gb=6.0)
    check_vram(Config(device="cuda:1", min_free_vram_gb=3.0))
    assert calls["device"] == "cuda:1"


def test_cache_hit_clears_last_warnings(make_synth):
    s, _ = make_synth(runtime=FakeRuntime(["bad thing"]))
    s.synthesize("hi")
    assert s.last_warnings == ["bad thing"]
    s.synthesize("hi")
    assert s.last_warnings == []
