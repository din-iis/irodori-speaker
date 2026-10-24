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
    tts_text,
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
    assert rt.calls[0].duration_scale == 0.8
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


@pytest.fixture
def fake_upstream(monkeypatch, tmp_path):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class RuntimeKey:
        checkpoint: str
        model_device: str
        codec_device: str
        model_precision: str
        codec_precision: str
        codec_repo: str

    cache = {}
    mod = types.ModuleType("irodori_tts.inference_runtime")
    mod.RuntimeKey = RuntimeKey

    order = []

    def clear_cached_runtime():
        order.append("clear")
        cache.clear()

    mod.clear_cached_runtime = clear_cached_runtime

    def get_cached_runtime(key):
        order.append("get")
        if key not in cache:
            cache.clear()  # 上流は key が変わると旧 runtime を unload する
            cache[key] = types.SimpleNamespace(
                key=key, model_cfg=types.SimpleNamespace(use_speaker_condition_resolved=True)
            )
        return cache[key], True

    mod.get_cached_runtime = get_cached_runtime
    monkeypatch.setitem(sys.modules, "irodori_tts", types.ModuleType("irodori_tts"))
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", mod)
    monkeypatch.setattr(synth, "_LOADED_KEY", None, raising=False)
    vram_calls = []
    monkeypatch.setattr(
        synth, "check_vram", lambda cfg: (order.append("vram"), vram_calls.append(cfg.checkpoint))
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    return types.SimpleNamespace(order=order, vram_calls=vram_calls, ckpt=str(ckpt), tmp=tmp_path)


def _real_synth(fake, checkpoint=None, name="a"):
    voice = fake.tmp / "v.wav"
    voice.write_bytes(b"abc")
    cfg = Config(device="cpu", checkpoint=checkpoint or fake.ckpt)
    return Synthesizer(cfg, fake.tmp / f"seg{name}", [voice], fake.tmp / "lat")


def test_second_synthesizer_with_same_key_skips_vram_check(fake_upstream, monkeypatch):
    first = _real_synth(fake_upstream, name="1")
    first.runtime

    def boom(cfg):
        raise RuntimeError("VRAM 不足")

    monkeypatch.setattr(synth, "check_vram", boom)
    second = _real_synth(fake_upstream, name="2")
    assert second.runtime is first.runtime


def test_runtime_follows_upstream_cache_after_switch(fake_upstream):
    other = fake_upstream.tmp / "ckpt2"
    other.mkdir()
    first = _real_synth(fake_upstream, name="1")
    old = first.runtime
    _real_synth(fake_upstream, checkpoint=str(other), name="2").runtime
    again = first.runtime
    assert again is not old
    assert again.key.checkpoint == fake_upstream.ckpt


def test_vram_checked_when_key_changes(fake_upstream):
    other = fake_upstream.tmp / "ckpt2"
    other.mkdir()
    _real_synth(fake_upstream, name="1").runtime
    assert fake_upstream.vram_calls == [fake_upstream.ckpt]
    _real_synth(fake_upstream, checkpoint=str(other), name="2").runtime
    assert fake_upstream.vram_calls == [fake_upstream.ckpt, str(other)]


def test_first_load_does_not_clear(fake_upstream):
    _real_synth(fake_upstream).runtime
    assert fake_upstream.order == ["vram", "get"]


def test_same_key_calls_neither_clear_nor_vram(fake_upstream):
    s = _real_synth(fake_upstream)
    s.runtime
    fake_upstream.order.clear()
    s.runtime
    assert fake_upstream.order == ["get"]


def test_key_change_clears_before_vram_check_and_load(fake_upstream):
    other = fake_upstream.tmp / "ckpt2"
    other.mkdir()
    _real_synth(fake_upstream, name="1").runtime
    fake_upstream.order.clear()
    _real_synth(fake_upstream, checkpoint=str(other), name="2").runtime
    assert fake_upstream.order == ["clear", "vram", "get"]


def test_loaded_key_reset_when_load_fails(fake_upstream, monkeypatch):
    s = _real_synth(fake_upstream)
    s.runtime

    def fail(key):
        raise RuntimeError("OOM")

    monkeypatch.setattr(sys.modules["irodori_tts.inference_runtime"], "get_cached_runtime", fail)
    other = fake_upstream.tmp / "ckpt2"
    other.mkdir()
    with pytest.raises(RuntimeError):
        _real_synth(fake_upstream, checkpoint=str(other), name="2").runtime
    assert synth._LOADED_KEY is None


def _fake_upstream(monkeypatch):
    calls = {"clear": 0, "get": 0}

    class Key(types.SimpleNamespace):
        def __eq__(self, other):
            return vars(self) == vars(other)

    def get_cached_runtime(key):
        calls["get"] += 1
        rt = types.SimpleNamespace(model_cfg=types.SimpleNamespace(use_speaker_condition_resolved=True))
        return rt, True

    def clear_cached_runtime():
        calls["clear"] += 1

    mod = types.ModuleType("irodori_tts.inference_runtime")
    mod.RuntimeKey = lambda **kw: Key(**kw)
    mod.get_cached_runtime = get_cached_runtime
    mod.clear_cached_runtime = clear_cached_runtime
    monkeypatch.setitem(sys.modules, "irodori_tts", types.ModuleType("irodori_tts"))
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", mod)
    monkeypatch.setattr(synth, "_LOADED_KEY", None)
    return calls


def test_release_runtime_clears_upstream_and_rechecks_vram(monkeypatch):
    calls = _fake_upstream(monkeypatch)
    checks = []
    monkeypatch.setattr(synth, "check_vram", lambda cfg: checks.append(1))
    cfg = Config(device="cpu")
    synth.load_runtime(cfg, "ckpt")
    synth.load_runtime(cfg, "ckpt")
    assert len(checks) == 1  # 同じキーでは再確認しない
    synth.release_runtime()
    assert calls["clear"] == 1
    assert synth._LOADED_KEY is None
    synth.load_runtime(cfg, "ckpt")
    assert len(checks) == 2  # 解放後は読み直しなので VRAM を再確認する


def test_release_runtime_without_irodori_tts_is_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "irodori_tts", None)
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", None)
    monkeypatch.setattr(synth, "_LOADED_KEY", None)
    synth.release_runtime()  # 例外にならない
    assert synth._LOADED_KEY is None


@pytest.mark.parametrize(
    "src,expected",
    [
        ("こんにちは。", "こんにちは"),
        ("「行こう。」", "「行こう」"),
        ("本当？", "本当？"),
        ("😊そうだね。", "😊そうだね"),
        ("……。", "……"),
        ("。", "。"),
        ("わかった。 ", "わかった"),
        ("ええ、", "ええ、"),
        ("やった！」", "やった！」"),
        ("おはよう．．", "おはよう"),
    ],
)
def test_tts_text(src, expected):
    assert tts_text(src) == expected


def test_request_strips_final_period(make_synth):
    s, _ = make_synth()
    assert s._request("こんにちは。").text == "こんにちは"


def test_synthesize_passes_stripped_text_to_runtime(make_synth):
    s, rt = make_synth()
    s.synthesize("こんにちは。")
    assert rt.calls[0].text == "こんにちは"


def test_cache_key_includes_text_rule(monkeypatch):
    cfg = Config(device="cpu")
    before = cache_key(cfg, "hi", "L")
    monkeypatch.setattr(synth, "TEXT_RULE", "other-rule")
    assert cache_key(cfg, "hi", "L") != before
