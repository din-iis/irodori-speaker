import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from irodori_speaker import synth
from irodori_speaker.config import Config, load_config
from irodori_speaker.lora import lora_key, normalize_lora, resolve_lora_adapter
from irodori_speaker.pipeline import chapter_signature
from irodori_speaker.synth import Synthesizer, cache_key


def make_adapter(root: Path, name="ad", weights="adapter_model.safetensors", body=b"W1", config=b"{}") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_bytes(config)
    if weights:
        (d / weights).write_bytes(body)
    return d


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def synthesize(self, req):
        self.calls.append(req)
        return types.SimpleNamespace(audio="AUDIO", sample_rate=48000, messages=[])


@pytest.fixture
def req_synth(tmp_path, monkeypatch):
    """_request が組む SamplingRequest の引数を見るための Synthesizer（runtime は偽）。"""
    voice = tmp_path / "v.wav"
    voice.write_bytes(b"abc")
    monkeypatch.setattr(synth, "_sampling_request", lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(synth, "_save_wav", lambda path, audio, sr: Path(path).write_bytes(b"RIFF"))
    monkeypatch.setattr(Synthesizer, "latent_path", property(lambda self: tmp_path / "x.pt"))

    def make(cfg=None, voices=True):
        cfg = cfg or Config(device="cpu")
        return Synthesizer(cfg, tmp_path / "seg", [voice] if voices else [], tmp_path / "lat")

    return make


# ---- normalize / resolve / key ----


@pytest.mark.parametrize("value", ["none", "off", "base", "NONE", " Off ", None, "", "  "])
def test_normalize_lora_はアダプタなしを_None_に寄せる(value):
    assert normalize_lora(value) is None


def test_normalize_lora_は通常の値を整えて返す():
    assert normalize_lora(" adapters/x ") == "adapters/x"


def test_resolve_は_None_なら_None(tmp_path):
    assert resolve_lora_adapter(None) is None
    assert resolve_lora_adapter("none") is None


def test_resolve_はディレクトリが無ければ拒否する(tmp_path):
    with pytest.raises(ValueError, match="LoRA"):
        resolve_lora_adapter(str(tmp_path / "nope"))


def test_resolve_は_adapter_config_が無ければ拒否する(tmp_path):
    d = make_adapter(tmp_path)
    (d / "adapter_config.json").unlink()
    with pytest.raises(ValueError, match="adapter_config.json"):
        resolve_lora_adapter(str(d))


def test_resolve_は重みが無ければ拒否する(tmp_path):
    d = make_adapter(tmp_path, weights=None)
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        resolve_lora_adapter(str(d))


@pytest.mark.parametrize("weights", ["adapter_model.safetensors", "adapter_model.bin"])
def test_resolve_は_safetensors_か_bin_を受け付ける(tmp_path, weights):
    d = make_adapter(tmp_path, weights=weights)
    assert resolve_lora_adapter(str(d)) == d


def test_resolve_はチルダを展開する(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = make_adapter(tmp_path)
    assert resolve_lora_adapter("~/ad") == d


def test_lora_key_は中身で変わり_mtime_では変わらない(tmp_path):
    d = make_adapter(tmp_path)
    k = lora_key(d)
    for f in d.iterdir():
        os.utime(f, (1, 1))
    assert lora_key(d) == k
    (d / "adapter_model.safetensors").write_bytes(b"W2")
    assert lora_key(d) != k
    k2 = lora_key(d)
    (d / "adapter_config.json").write_bytes(b'{"r": 8}')
    assert lora_key(d) != k2


def test_lora_key_は同じ中身なら場所が違っても同じ(tmp_path):
    a = make_adapter(tmp_path, "a")
    b = make_adapter(tmp_path, "b")
    assert lora_key(a) == lora_key(b)


# ---- cache_key ----


def test_cache_key_は_lora_が無ければ従来と同じ():
    cfg = Config()
    assert cache_key(cfg, "あ", "L", lora_key=None) == cache_key(cfg, "あ", "L")
    assert cache_key(cfg, "あ", "L", lora_key="K") != cache_key(cfg, "あ", "L")
    assert cache_key(cfg, "あ", "L", lora_key="K") != cache_key(cfg, "あ", "L", lora_key="J")


# ---- Synthesizer ----


def test_検証は_runtime_に触る前に行う(tmp_path, monkeypatch):
    order = []
    mod = types.ModuleType("irodori_tts.inference_runtime")
    mod.RuntimeKey = lambda **kw: types.SimpleNamespace(**kw)
    mod.clear_cached_runtime = lambda: order.append("clear")
    mod.get_cached_runtime = lambda key: (order.append("get"), None)[1]
    monkeypatch.setitem(sys.modules, "irodori_tts", types.ModuleType("irodori_tts"))
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", mod)
    monkeypatch.setattr(synth, "check_vram", lambda cfg: order.append("vram"))
    cfg = Config(device="cpu", lora_adapter=str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="LoRA"):
        Synthesizer(cfg, tmp_path / "seg", [], tmp_path / "lat")
    assert order == []


def test_request_は_lora_があるときだけ_lora_adapter_を渡す(req_synth, tmp_path):
    d = make_adapter(tmp_path)
    plain = req_synth()._request("あ")
    assert not hasattr(plain, "lora_adapter")
    s = req_synth(Config(device="cpu", lora_adapter=str(d)))
    assert s._request("あ").lora_adapter == str(d)
    assert req_synth(Config(device="cpu", lora_adapter=str(d)), voices=False)._request("あ").lora_adapter == str(d)


def test_lora_none_は無効として扱う(req_synth):
    s = req_synth(Config(device="cpu", lora_adapter="none"))
    assert s.lora_path is None and s.lora_key is None
    assert not hasattr(s._request("あ"), "lora_adapter")


def test_synthesize_のキーに_lora_が入る(req_synth, tmp_path, monkeypatch):
    d = make_adapter(tmp_path)
    s = req_synth(Config(device="cpu", lora_adapter=str(d)))
    rt = FakeRuntime()
    monkeypatch.setattr(Synthesizer, "runtime", property(lambda self: rt))
    path = s.synthesize("あ")
    assert path.stem == cache_key(s.cfg, "あ", s.latent_key, lora_key=lora_key(d))
    assert path.stem != cache_key(s.cfg, "あ", s.latent_key)


# ---- load_runtime ----


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    order = []
    mod = types.ModuleType("irodori_tts.inference_runtime")
    mod.RuntimeKey = lambda **kw: tuple(sorted(kw.items()))
    mod.clear_cached_runtime = lambda: order.append("clear")

    def get_cached_runtime(key):
        order.append("get")
        return types.SimpleNamespace(model_cfg=types.SimpleNamespace(use_speaker_condition_resolved=True)), True

    mod.get_cached_runtime = get_cached_runtime
    monkeypatch.setitem(sys.modules, "irodori_tts", types.ModuleType("irodori_tts"))
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", mod)
    monkeypatch.setattr(synth, "_LOADED_KEY", None)
    monkeypatch.setattr(synth, "check_vram", lambda cfg: order.append("vram"))
    return order


def test_load_runtime_はアダプタが変わると読み直す(upstream):
    cfg = Config(device="cpu")
    synth.load_runtime(cfg, "ck", lora_key="A")
    upstream.clear()
    synth.load_runtime(cfg, "ck", lora_key="B")
    assert upstream == ["clear", "vram", "get"]


def test_load_runtime_はアダプタを外しても読み直す(upstream):
    cfg = Config(device="cpu")
    synth.load_runtime(cfg, "ck", lora_key="A")
    upstream.clear()
    synth.load_runtime(cfg, "ck", lora_key=None)
    assert upstream == ["clear", "vram", "get"]


def test_load_runtime_は同じ組なら_clear_しない(upstream):
    cfg = Config(device="cpu")
    synth.load_runtime(cfg, "ck", lora_key="A")
    upstream.clear()
    synth.load_runtime(cfg, "ck", lora_key="A")
    assert "clear" not in upstream and "vram" not in upstream


def test_runtime_プロパティは_lora_key_を渡す(tmp_path, monkeypatch):
    d = make_adapter(tmp_path)
    seen = {}
    monkeypatch.setattr(synth, "resolve_checkpoint", lambda c: "ck")
    monkeypatch.setattr(synth, "load_runtime", lambda cfg, ck=None, lora_key=None: seen.update(lora_key=lora_key))
    s = Synthesizer(Config(device="cpu", lora_adapter=str(d)), tmp_path / "seg", [], tmp_path / "lat")
    s.runtime
    assert seen["lora_key"] == lora_key(d)


# ---- 章署名 ----


def test_署名はアダプタがあるときだけ変わる(req_synth, tmp_path):
    rows = [{"para": "0", "seg": "0", "text": "あ"}]
    cfg = Config()
    plain = req_synth()
    legacy = types.SimpleNamespace(latent_key=plain.latent_key)
    base = chapter_signature(rows, cfg, legacy)
    assert chapter_signature(rows, cfg, plain) == base
    d = make_adapter(tmp_path)
    withlora = req_synth(Config(device="cpu", lora_adapter=str(d)))
    sig = chapter_signature(rows, cfg, withlora)
    assert sig != base
    (d / "adapter_model.safetensors").write_bytes(b"W2")
    assert chapter_signature(rows, cfg, req_synth(Config(device="cpu", lora_adapter=str(d)))) != sig


# ---- CLI ----


def test_cli_は_lora_adapter_を_config_に反映する(tmp_path, monkeypatch):
    from irodori_speaker import cli

    seen = {}

    def fake_make(cfg, out_dir, voice, checkpoint):
        seen["lora"] = cfg.lora_adapter
        raise ValueError("stop")

    monkeypatch.setattr(cli, "_make_synthesizer", fake_make)
    monkeypatch.setattr(cli, "load_book", lambda out_dir: ({}, []))
    base = Config(device="cpu", lora_adapter="cfg-dir", output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(cli, "load_config", lambda path: base)
    novel = tmp_path / "n.txt"
    novel.write_text("あ", encoding="utf-8")

    for given, expected in (("my/dir", "my/dir"), ("none", None), (None, "cfg-dir")):
        argv = ["synth", str(novel), "--voice", "none"] + (["--lora-adapter", given] if given else [])
        cli.main(argv)
        assert seen["lora"] == expected


def test_cli_の_help_は日本語():
    from irodori_speaker import cli

    with pytest.raises(SystemExit):
        cli.main(["synth", "--help"])


def test_make_synthesizer_は_lora_を表示する(tmp_path, capsys):
    from irodori_speaker import cli

    d = make_adapter(tmp_path)
    cfg = Config(device="cpu", lora_adapter=str(d), output_dir=str(tmp_path / "out"))
    cli._make_synthesizer(cfg, tmp_path / "out" / "n", "none", None)
    assert f"話者 LoRA: {d}" in capsys.readouterr().out


# ---- WebUI ----


def test_Engine_key_for_は_lora_を_5_要素目に持つ(tmp_path):
    from irodori_speaker.webui import Engine

    engine = Engine(Config(lora_adapter="cfg-dir"))
    assert engine.key_for(None, None, None, "v")[4] == "cfg-dir"  # 空欄は config
    assert engine.key_for(None, None, None, "v", "")[4] == "cfg-dir"
    assert engine.key_for(None, None, None, "v", " mine ")[4] == "mine"
    assert engine.key_for(None, None, None, "v", "none")[4] is None  # none で config を打ち消す
    assert Engine(Config()).key_for(None, None, None, "v")[4] is None


def test_Engine_cfg_for_は_lora_adapter_を反映する():
    from irodori_speaker.webui import Engine

    engine = Engine(Config(lora_adapter="cfg-dir"))
    assert engine.cfg_for(engine.key_for(None, None, None, "v", "x")).lora_adapter == "x"
    assert engine.cfg_for(engine.key_for(None, None, None, "v", "none")).lora_adapter is None


def test_Engine_はアダプタが変わると_Synthesizer_を作り直す(tmp_path):
    from irodori_speaker.webui import Engine

    a = make_adapter(tmp_path, "a")
    b = make_adapter(tmp_path, "b", body=b"W2")
    cfg = replace(Config(device="cpu"), output_dir=str(tmp_path / "out"))
    engine = Engine(cfg)
    k1 = engine.key_for(None, None, None, "none", str(a))
    k2 = engine.key_for(None, None, None, "none", str(b))
    s1 = engine.get(k1, tmp_path / "seg")
    assert engine.get(k1, tmp_path / "seg") is s1
    s2 = engine.get(k2, tmp_path / "seg")
    assert s2 is not s1 and s2.lora_path == b


def _handlers(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    from irodori_speaker import webui

    monkeypatch.setattr(synth, "release_runtime", lambda: None)
    app = webui.build_app(replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out")))
    return {f.fn.__name__: f.fn for f in app.fns.values()}, app


def test_generate_は末尾の_lora_引数を取り位置引数の呼び出しも通る(monkeypatch, tmp_path):
    from irodori_speaker import webui

    fns, _ = _handlers(monkeypatch, tmp_path)
    keys = []

    def fake_key_for(self, *a):
        keys.append(a)
        raise ValueError("stop")

    monkeypatch.setattr(webui.Engine, "key_for", fake_key_for)
    base = ("本文", None, None, None, "narrator", None, 3.0, 5.0, 0.8, 42)
    with pytest.raises(Exception):
        next(fns["generate"](*base))
    with pytest.raises(Exception):
        next(fns["generate"](*base, "my/dir"))
    assert keys[0] == (None, None, None, "narrator", None)
    assert keys[1] == (None, None, None, "narrator", "my/dir")


def test_画面の_lora_欄は各_click_の_inputs_末尾にある(monkeypatch, tmp_path):
    _, app = _handlers(monkeypatch, tmp_path)
    lora_boxes = [c for c in app.blocks.values() if getattr(c, "label", "") and "LoRA" in str(c.label)]
    assert lora_boxes, "LoRA の Textbox が無い"
    lora = lora_boxes[0]
    names = {"load_model", "generate", "compare_emotions", "run_synth", "run_all"}
    seen = set()
    for f in app.fns.values():
        if f.fn.__name__ in names:
            assert f.inputs[-1] is lora
            seen.add(f.fn.__name__)
    assert seen == names
