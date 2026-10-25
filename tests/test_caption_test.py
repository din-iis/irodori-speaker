import importlib.util
import sys
import types
from pathlib import Path

from irodori_speaker import synth

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "caption_test.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("caption_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, monkeypatch, *extra):
    # 上流の SamplingRequest と save_wav だけを差し替え、Synthesizer の組み立てはそのまま通す
    upstream = types.ModuleType("irodori_tts.inference_runtime")
    upstream.SamplingRequest = lambda **kw: types.SimpleNamespace(**kw)
    upstream.save_wav = lambda path, audio, sr: Path(path).write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "irodori_tts", types.ModuleType("irodori_tts"))
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", upstream)

    requests, checkpoints = [], []

    class FakeRuntime:
        def synthesize(self, req):
            requests.append(req)
            return types.SimpleNamespace(audio=None, sample_rate=24000, messages=[])

    def runtime(self):
        checkpoints.append(self.cfg.checkpoint)
        return FakeRuntime()

    monkeypatch.setattr(synth.Synthesizer, "runtime", property(runtime))
    monkeypatch.setattr(synth.Synthesizer, "latent_path", property(lambda self: tmp_path / "lat.pt"))

    voice = tmp_path / "v.wav"
    voice.write_bytes(b"abc")
    config = tmp_path / "config.yaml"
    config.write_text(f"voice: {voice}\noutput_dir: {tmp_path / 'out'}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv",
        ["caption_test.py", "-c", str(config), "--checkpoint", "org/Base-v4.1",
         "--checkpoint", "/x/merged.safetensors", *extra],
    )
    mod = _load_script()
    mod.main()
    return mod, requests, checkpoints, tmp_path / "out" / "caption_test"


def test_caption_test_passes_caption_per_checkpoint(tmp_path, monkeypatch):
    mod, requests, checkpoints, out = _run(tmp_path, monkeypatch)

    # 1 文あたり: キャプションなし 1 + 感情ごとに（キャプション × SCALES + 絵文字 1）
    per_ckpt = len(mod.TEXTS) * (1 + (len(mod.CAPTIONS) - 1) * (len(mod.SCALES) + 1))
    assert len(requests) == 2 * per_ckpt
    assert set(checkpoints) == {"org/Base-v4.1", "/x/merged.safetensors"}
    angry = [r for r in requests if r.caption == mod.CAPTIONS["angry"]]
    assert sorted(r.cfg_scale_caption for r in angry) == [3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0]
    assert requests[0].caption is None
    assert all(not r.text.endswith("。") and r.seed == 42 for r in requests)
    assert all(r.ref_latent and not getattr(r, "no_ref", False) for r in requests)
    # 絵文字の行は emotion_emoji の強度 3 を文頭に付け、キャプションは渡さない
    emoji = [r for r in requests if r.text.startswith("😠😠")]
    assert len(emoji) == 2 * len(mod.TEXTS) and all(r.caption is None for r in emoji)

    assert (out / "0_Base-v4.1" / "1_none.wav").exists()
    assert (out / "0_Base-v4.1" / "1_angry_emoji.wav").exists()
    assert (out / "1_merged" / "2_whisper_cap5.wav").exists()


def test_caption_test_no_ref_drops_reference(tmp_path, monkeypatch):
    _, requests, _, out = _run(tmp_path, monkeypatch, "--no-ref")

    assert requests and all(r.ref_latent is None and r.no_ref for r in requests)
    assert (out / "0_Base-v4.1_noref" / "1_none.wav").exists()
    assert (out / "1_merged_noref" / "2_happy_cap3.wav").exists()
