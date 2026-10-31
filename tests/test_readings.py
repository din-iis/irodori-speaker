import types
from dataclasses import replace
from pathlib import Path

import pytest

from irodori_speaker import synth
from irodori_speaker.config import Config, load_config
from irodori_speaker.pipeline import chapter_signature
from irodori_speaker.readings import apply_readings, load_readings, readings_key
from irodori_speaker.synth import Synthesizer, cache_key, segment_cache_key


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

    def fake_save(path, audio, sr):
        Path(path).write_bytes(b"RIFF")

    monkeypatch.setattr(synth, "_sampling_request", lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(synth, "_save_wav", fake_save)

    def make(cfg=None, runtime=None, readings=None):
        cfg = cfg or Config(device="cpu")
        s = Synthesizer(cfg, tmp_path / "seg", [voice], tmp_path / "lat", tmp_path / "warn.tsv",
                        readings=readings)
        rt = runtime or FakeRuntime()
        monkeypatch.setattr(Synthesizer, "runtime", property(lambda self: rt))
        monkeypatch.setattr(Synthesizer, "latent_path", property(lambda self: tmp_path / "x.pt"))
        return s, rt

    return make


def write_yaml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---- apply_readings / readings_key ----


def test_最長一致を優先して置換する():
    table = {"紫陽花": "あじさい", "紫": "むらさき"}
    assert apply_readings("紫陽花と紫", table) == "あじさいとむらさき"


def test_置換は連鎖しない():
    table = {"花": "はな", "はな": "ハナ"}
    assert apply_readings("花とはな", table) == "はなとハナ"


def test_絵文字入りの本文でも置換できる():
    assert apply_readings("😊紫陽花が咲く", {"紫陽花": "あじさい"}) == "😊あじさいが咲く"


def test_辞書が空なら本文をそのまま返す():
    text = "そのまま。"
    assert apply_readings(text, {}) is text
    assert apply_readings(text, None) is text


def test_readings_key_は順序に依存せず空なら_None():
    assert readings_key({"a": "b", "c": "d"}) == readings_key({"c": "d", "a": "b"})
    assert readings_key({"a": "b"}) != readings_key({"a": "c"})
    assert readings_key({}) is None


# ---- load_readings ----


def test_config_の辞書を読む(tmp_path):
    f = write_yaml(tmp_path / "r.yaml", "紫陽花: あじさい\n")
    cfg = Config(readings=[str(f)])
    assert load_readings(cfg, None) == {"紫陽花": "あじさい"}


def test_本ごとの辞書が後勝ちで_config_の複数ファイルも後勝ち(tmp_path):
    a = write_yaml(tmp_path / "a.yaml", "花: はな\n葉: は\n")
    b = write_yaml(tmp_path / "b.yaml", "花: か\n")
    book = tmp_path / "out" / "novel"
    write_yaml(book / "readings.yaml", "葉: よう\n")
    table = load_readings(Config(readings=[str(a), str(b)]), book)
    assert table == {"花": "か", "葉": "よう"}


def test_本ごとの辞書が無ければ_config_だけ(tmp_path):
    f = write_yaml(tmp_path / "r.yaml", "花: はな\n")
    book = tmp_path / "out" / "novel"
    book.mkdir(parents=True)
    assert load_readings(Config(readings=[str(f)]), book) == {"花": "はな"}
    assert load_readings(Config(), book) == {}
    assert load_readings(Config(), None) == {}


def test_空のファイルは空の辞書(tmp_path):
    f = write_yaml(tmp_path / "r.yaml", "")
    assert load_readings(Config(readings=[str(f)]), None) == {}


def test_チルダを展開する(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_yaml(tmp_path / "r.yaml", "花: はな\n")
    assert load_readings(Config(readings=["~/r.yaml"]), None) == {"花": "はな"}


def test_辞書ファイルが無ければ拒否する(tmp_path):
    with pytest.raises(ValueError, match="読み辞書が見つかりません"):
        load_readings(Config(readings=[str(tmp_path / "nope.yaml")]), None)


@pytest.mark.parametrize("body", ["- a\n- b\n", "1: いち\n", "花: 1\n", "花: [a]\n", "just text\n"])
def test_不正な書式は日本語で拒否する(tmp_path, body):
    f = write_yaml(tmp_path / "r.yaml", body)
    with pytest.raises(ValueError, match="読み辞書"):
        load_readings(Config(readings=[str(f)]), None)


def test_本ごとの辞書の不正も拒否する(tmp_path):
    book = tmp_path / "novel"
    write_yaml(book / "readings.yaml", "- a\n")
    with pytest.raises(ValueError, match="読み辞書"):
        load_readings(Config(), book)


def test_readings_がリストでなければ_config_が拒否する(tmp_path):
    for bad in ("readings: a.yaml\n", "readings: [1]\n", "readings: {a: b}\n"):
        f = write_yaml(tmp_path / "c.yaml", bad)
        with pytest.raises(ValueError, match="readings"):
            load_config(f)
    f = write_yaml(tmp_path / "c.yaml", "readings: [a.yaml]\n")
    assert load_config(f).readings == ["a.yaml"]


# ---- キー ----


def test_辞書で本文が変わる文だけキーが変わる():
    cfg = Config()
    table = {"紫陽花": "あじさい"}
    changed = ("紫陽花が咲く", "紫陽花が咲く")
    assert segment_cache_key(cfg, changed[0], "L", readings=table) != cache_key(cfg, changed[1], "L")
    assert segment_cache_key(cfg, "花が咲く", "L", readings=table) == cache_key(cfg, "花が咲く", "L")
    assert segment_cache_key(cfg, "紫陽花が咲く", "L", readings=table) == cache_key(cfg, "あじさいが咲く", "L")


def test_segment_cache_key_は辞書なしなら従来のキーと同じ():
    cfg = Config()
    assert segment_cache_key(cfg, "あ", "L") == cache_key(cfg, "あ", "L")
    assert segment_cache_key(cfg, "あ", "L", readings={}) == cache_key(cfg, "あ", "L")


# ---- Synthesizer ----


def test_synthesize_は置換後の本文でモデルを呼びキャッシュする(make_synth):
    s, rt = make_synth(readings={"紫陽花": "あじさい"})
    path = s.synthesize("紫陽花が咲く。")
    assert rt.calls[0].text == "あじさいが咲く"  # 置換は 1 回だけ、その後に tts_text
    assert path.name == f"{s.segment_key('紫陽花が咲く。')}.wav"
    assert path.read_bytes() == b"RIFF"
    s.synthesize("紫陽花が咲く。")
    assert len(rt.calls) == 1


def test_辞書を変えると本文が変わる文だけ再合成する(make_synth):
    s, rt = make_synth(readings={"紫陽花": "あじさい"})
    s.synthesize("紫陽花")
    s.synthesize("桜")
    s.readings = {"紫陽花": "しようか"}
    s.synthesize("紫陽花")
    s.synthesize("桜")
    assert [c.text for c in rt.calls] == ["あじさい", "桜", "しようか"]


def test_warnings_log_は置換前の本文(make_synth, tmp_path):
    s, _ = make_synth(runtime=FakeRuntime(["bad"]), readings={"紫陽花": "あじさい"})
    path = s.synthesize("紫陽花")
    line = (tmp_path / "warn.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert line == f"{path.stem}\t紫陽花\tbad"


def test_generate_once_にも辞書が効く(make_synth):
    import numpy as np

    class Audio:
        def detach(self): return self
        def cpu(self): return self
        def float(self): return self
        def numpy(self): return np.ones((1, 4))

    rt = FakeRuntime()
    rt.synthesize = lambda req: (rt.calls.append(req), types.SimpleNamespace(
        audio=Audio(), sample_rate=24000, messages=[]))[1]
    s, _ = make_synth(runtime=rt, readings={"紫陽花": "あじさい"})
    s.generate_once("紫陽花")
    assert rt.calls[0].text == "あじさい"


def test_readings_key_プロパティは辞書が空なら_None(make_synth):
    s, _ = make_synth()
    assert s.readings_key is None
    s.readings = {"a": "b"}
    assert s.readings_key == readings_key({"a": "b"})


# ---- 章署名 ----


def _rows():
    return [{"para": "0", "seg": "0", "text": "紫陽花"}]


def test_署名は辞書があるときだけ変わり無ければ従来と同じ(make_synth):
    cfg = Config()
    s, _ = make_synth()
    legacy = types.SimpleNamespace(latent_key=s.latent_key)  # この機能より前の synth 相当
    base = chapter_signature(_rows(), cfg, legacy)
    assert chapter_signature(_rows(), cfg, s) == base
    s.readings = {"紫陽花": "あじさい"}
    with_dict = chapter_signature(_rows(), cfg, s)
    assert with_dict != base
    s.readings = {"紫陽花": "しようか"}
    assert chapter_signature(_rows(), cfg, s) != with_dict


# ---- CLI / WebUI ----


def test_make_synthesizer_は本ごとの辞書を渡す(tmp_path, capsys):
    from irodori_speaker import cli

    book = tmp_path / "out" / "novel"
    write_yaml(book / "readings.yaml", "紫陽花: あじさい\n")
    cfg = replace(Config(device="cpu"), output_dir=str(tmp_path / "out"))
    s = cli._make_synthesizer(cfg, book, "none", None)
    assert s.readings == {"紫陽花": "あじさい"}
    assert "読み辞書: 1 語" in capsys.readouterr().out


def test_Engine_get_は_out_dir_ごとに辞書を差し替える(tmp_path):
    from irodori_speaker.webui import Engine

    a, b = tmp_path / "out" / "a", tmp_path / "out" / "b"
    write_yaml(a / "readings.yaml", "花: はな\n")
    cfg = replace(Config(device="cpu"), output_dir=str(tmp_path / "out"), voices_dir=str(tmp_path / "v"))
    engine = Engine(cfg)
    key = engine.key_for(None, None, None, "none")
    s1 = engine.get(key, a / "segments", out_dir=a)
    assert s1.readings == {"花": "はな"}
    s2 = engine.get(key, b / "segments", out_dir=b)
    assert s2 is s1 and s2.readings == {}
    s3 = engine.get(key, tmp_path / "preview")  # 試聴は config の辞書だけ
    assert s3.readings == {}


def test_YAML_の構文エラーは日本語の_ValueError_でパスを含む(tmp_path):
    f = write_yaml(tmp_path / "r.yaml", "花: [はな\n")
    with pytest.raises(ValueError, match="読み辞書") as e:
        load_readings(Config(readings=[str(f)]), None)
    assert str(f) in str(e.value)
    book = tmp_path / "novel"
    b = write_yaml(book / "readings.yaml", "花: [はな\n")
    with pytest.raises(ValueError) as e:
        load_readings(Config(), book)
    assert str(b) in str(e.value)
