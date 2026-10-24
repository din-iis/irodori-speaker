import pytest

from irodori_speaker.config import load_config
from irodori_speaker.pipeline import TSV_FIELDS
from irodori_speaker.webui import (
    CHECKPOINT_CHOICES,
    Engine,
    add_voice,
    dataframe_rows,
    emoji_prefixed,
    llm_overrides_from_ui,
    parse_seed,
    parse_steps,
    rows_from_dataframe,
    should_reload,
    stream_progress,
)


@pytest.mark.parametrize("value", [None, "", "  ", -1, -1.0, "-1", -5])
def test_parse_seed_はランダム指定を_None_にする(value):
    assert parse_seed(value) is None


@pytest.mark.parametrize("value, expected", [(0, 0), (42, 42), (42.0, 42), ("7", 7)])
def test_parse_seed_は整数にする(value, expected):
    assert parse_seed(value) == expected


@pytest.mark.parametrize("value", [None, "", 0, "  "])
def test_parse_steps_は空なら既定にする(value):
    assert parse_steps(value) is None


@pytest.mark.parametrize("value, expected", [(40, 40), (40.0, 40), ("12", 12)])
def test_parse_steps_は整数にする(value, expected):
    assert parse_steps(value) == expected


def test_llm_overrides_from_ui_は5項目を渡す():
    got = llm_overrides_from_ui("openai", " http://h:8080/v1 ", " m ", True, 3.0)
    assert got == {
        "api": "openai",
        "base_url": "http://h:8080/v1",
        "model": "m",
        "think": True,
        "concurrency": 3,
    }


def test_llm_overrides_from_ui_は空欄を上書きに含めない():
    got = llm_overrides_from_ui("ollama", "", None, False, None)
    assert got == {"api": "ollama", "think": False}


def test_llm_overrides_from_ui_は並列数を1以上にする():
    assert llm_overrides_from_ui("ollama", "u", "m", False, 0)["concurrency"] == 1


def _row(seg: int, text: str) -> dict:
    return {
        "chapter": "0",
        "para": "0",
        "seg": str(seg),
        "kind": "narration",
        "labels": "neutral:1",
        "text": text,
        "source": f"原文{seg}",
    }


def test_rows_from_dataframe_は_text_列だけ反映する():
    original = [_row(0, "あ"), _row(1, "い")]
    df_rows = [
        ["0", "0", "0", "dialogue", "happy:3", "😊あ", "改ざん"],
        ["0", "0", "1", "narration", "neutral:1", "い", "原文1"],
    ]
    got = rows_from_dataframe(df_rows, original)
    assert [r["text"] for r in got] == ["😊あ", "い"]
    assert got[0]["kind"] == "narration"
    assert got[0]["labels"] == "neutral:1"
    assert got[0]["source"] == "原文0"
    assert list(got[0]) == TSV_FIELDS


def test_rows_from_dataframe_は元の行を変更しない():
    original = [_row(0, "あ")]
    rows_from_dataframe([["0", "0", "0", "", "", "X", ""]], original)
    assert original[0]["text"] == "あ"


def test_rows_from_dataframe_は数値型のキーでも対応づける():
    original = [_row(0, "あ")]
    got = rows_from_dataframe([[0, 0.0, 0, "", "", "X", ""]], original)
    assert got[0]["text"] == "X"


def test_rows_from_dataframe_は欠損の_text_を空文字にする():
    original = [_row(0, "あ")]
    got = rows_from_dataframe([["0", "0", "0", "", "", float("nan"), ""]], original)
    assert got[0]["text"] == ""


def test_rows_from_dataframe_は_pandas_の_DataFrame_も受け付ける():
    pd = pytest.importorskip("pandas")
    original = [_row(0, "あ")]
    df = pd.DataFrame([["0", "0", "0", "", "", "X", ""]], columns=TSV_FIELDS)
    assert rows_from_dataframe(df, original)[0]["text"] == "X"


def test_rows_from_dataframe_は台本に無い行を拒否する():
    with pytest.raises(ValueError):
        rows_from_dataframe([["9", "9", "9", "", "", "X", ""]], [_row(0, "あ")])


def test_rows_from_dataframe_は行が足りなければ拒否する():
    with pytest.raises(ValueError):
        rows_from_dataframe([["0", "0", "0", "", "", "X", ""]], [_row(0, "あ"), _row(1, "い")])


def test_rows_from_dataframe_は元の行が空なら空を返す():
    assert rows_from_dataframe([], []) == []


@pytest.mark.parametrize(
    "text, emoji, expected",
    [
        ("こんにちは", "😊", "😊こんにちは"),
        ("「こんにちは」", "😊", "「😊こんにちは」"),
        ("『やあ』", "😲😲", "『😲😲やあ』"),
        ("（小声）", "👂", "（👂小声）"),
        ("こんにちは", "", "こんにちは"),
        ("", "😊", "😊"),
    ],
)
def test_emoji_prefixed(text, emoji, expected):
    assert emoji_prefixed(text, emoji) == expected


def test_checkpoint_choices_は既定を含む():
    assert CHECKPOINT_CHOICES == [
        "Aratako/Irodori-TTS-v4.1-Small",
        "Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only",
        "Aratako/Irodori-TTS-v4.1-Small-MF",
    ]


def test_webui_は_gradio_無しで_import_できる():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['gradio'] = None; sys.modules['torch'] = None;"
        "import irodori_speaker.webui"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_画面が起動する(tmp_path):
    pytest.importorskip("gradio")
    requests = pytest.importorskip("requests")
    from irodori_speaker.webui import build_app

    cfg = load_config("config.example.yaml")
    app = build_app(cfg)
    app.launch(prevent_thread_lock=True, server_port=7861)
    try:
        assert requests.get("http://127.0.0.1:7861/").status_code == 200
    finally:
        app.close()


def test_dataframe_rows_は_TSV_列順の配列にする():
    assert dataframe_rows([_row(0, "あ")]) == [["0", "0", "0", "narration", "neutral:1", "あ", "原文0"]]


def test_stream_progress_は進捗を流し戻り値を返す():
    def work(progress):
        progress(1, 2, "a")
        progress(2, 2, "b")
        return "結果"

    gen = stream_progress(work)
    seen = []
    try:
        while True:
            seen.append(next(gen))
    except StopIteration as stop:
        result = stop.value
    assert result == "結果"
    assert seen and seen[-1] == (2, 2, "b")


def test_stream_progress_は例外を呼び出し側に伝える():
    def work(progress):
        raise RuntimeError("壊れた")

    with pytest.raises(RuntimeError, match="壊れた"):
        for _ in stream_progress(work):
            pass


def test_add_voice_は声を_voices_にコピーする(tmp_path):
    src = tmp_path / "up" / "x.wav"
    src.parent.mkdir()
    src.write_bytes(b"RIFF")
    dest = add_voice(src, "narrator", tmp_path / "voices")
    assert dest == tmp_path / "voices" / "narrator.wav"
    assert dest.read_bytes() == b"RIFF"


@pytest.mark.parametrize("voice_id", ["", "  ", "a/b", "..", ".hidden", "a\\b"])
def test_add_voice_は不正な_ID_を拒否する(tmp_path, voice_id):
    src = tmp_path / "x.wav"
    src.write_bytes(b"RIFF")
    with pytest.raises(ValueError):
        add_voice(src, voice_id, tmp_path / "voices")


def test_add_voice_は既存の声を上書きしない(tmp_path):
    src = tmp_path / "x.wav"
    src.write_bytes(b"new")
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "narrator.wav").write_bytes(b"old")
    with pytest.raises(ValueError, match="既に"):
        add_voice(src, "narrator", voices)
    assert (voices / "narrator.wav").read_bytes() == b"old"


def test_add_voice_は音声ファイルが無ければ拒否する(tmp_path):
    with pytest.raises(ValueError):
        add_voice(None, "narrator", tmp_path / "voices")


def test_Engine_は_mark_unloaded_の後は未読み込みになる():
    engine = Engine(load_config(None))
    key = engine.key_for(None, None, None, "narrator")
    engine._key, engine._ready = key, True
    assert engine.is_loaded(key)
    engine.mark_unloaded()
    assert not engine.is_loaded(key)


def _handlers(monkeypatch, tmp_path):
    """build_app 内のハンドラを取り出す（release_runtime の呼び出しを記録する）。"""
    pytest.importorskip("gradio")
    from dataclasses import replace

    from irodori_speaker import synth, webui

    events = []
    monkeypatch.setattr(synth, "release_runtime", lambda: events.append("release"))
    app = webui.build_app(replace(load_config("config.example.yaml"), output_dir=str(tmp_path / "out")))
    fns = {f.fn.__name__: f.fn for f in app.fns.values()}
    return fns, events


def test_LLM_で絵文字を付ける前に_TTS_を解放する(monkeypatch, tmp_path):
    fns, events = _handlers(monkeypatch, tmp_path)
    import irodori_speaker.emotion as emotion

    class Boom:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            events.append("llm")
            raise RuntimeError("stop")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(emotion, "LLMServer", Boom)
    gen = fns["annotate_text"]("こんにちは", "ollama", "http://x", "m", False, 1)
    with pytest.raises(Exception):
        list(gen)
    assert events[:2] == ["release", "llm"]


def test_台本段階の前に_TTS_を解放する(monkeypatch, tmp_path):
    fns, events = _handlers(monkeypatch, tmp_path)
    from irodori_speaker import webui

    monkeypatch.setattr(webui, "build_script", lambda *a, **k: events.append("script") or [])
    novel = tmp_path / "n.txt"
    novel.write_text("第一章\n本文です。\n", encoding="utf-8")
    gen = fns["run_script"](None, str(novel), "", False, False, False, None, "cpu", "fp32", "v",
                            "ollama", "http://x", "m", False, 1, [], [])
    with pytest.raises(Exception):  # build_script をモックしたので load_book が失敗する
        list(gen)
    assert events[:2] == ["release", "script"]


def test_should_reload_は小説が変わったときだけ真(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    assert should_reload(a, None)
    assert not should_reload(a, str(a))
    assert should_reload(b, str(a))


def test_show_script_は同じ小説なら表を変えない(monkeypatch, tmp_path):
    fns, _events = _handlers(monkeypatch, tmp_path)
    from irodori_speaker.pipeline import book_dir, write_script

    novel = tmp_path / "n.txt"
    novel.write_text("第一章\n本文です。\n", encoding="utf-8")
    cfg = load_config("config.example.yaml")
    out_dir = book_dir(replace_output(cfg, tmp_path), novel)
    out_dir.mkdir(parents=True)
    (out_dir / "chapters.json").write_text('{"chapters": [{"idx": 0, "title": "t"}]}', encoding="utf-8")
    write_script(out_dir / "script.tsv", [_row(0, "あ")])

    table, rows, status, loaded = fns["show_script"](None, str(novel), None)
    assert rows and loaded == str(novel)
    again = fns["show_script"](None, str(novel), loaded)
    assert again[1] == {"__type__": "update"} and again[0] == {"__type__": "update"}
    assert again[3] == loaded

    other = tmp_path / "m.txt"
    other.write_text("本文", encoding="utf-8")
    assert fns["show_script"](None, str(other), loaded)[3] == str(other)


def replace_output(cfg, tmp_path):
    from dataclasses import replace

    return replace(cfg, output_dir=str(tmp_path / "out"))


# -- 試聴の分割合成 --

LONG = "あ" * 15 + "、" + "い" * 15 + "、" + "う" * 15 + "。"  # 48 文字で 1 文


def test_split_preview_text_は長い文を_max_chars_以内に分ける():
    from irodori_speaker.webui import split_preview_text

    paragraphs = split_preview_text(LONG, 40)
    assert len(paragraphs) == 1 and len(paragraphs[0]) >= 2
    assert all(len(s) <= 40 for s in paragraphs[0])
    assert "".join(paragraphs[0]) == LONG


def test_split_preview_text_は改行で段落を分け空行を無視する():
    from irodori_speaker.webui import split_preview_text

    assert split_preview_text("あ。い。\n\n  \nう。", 40) == [["あ。", "い。"], ["う。"]]


def test_split_preview_text_は短い文を1要素のままにする():
    from irodori_speaker.webui import split_preview_text

    assert split_preview_text("こんにちは。", 40) == [["こんにちは。"]]


def test_join_audio_は文の長さと無音の合計になる():
    import numpy as np

    from irodori_speaker.webui import join_audio

    a, b, c = (np.ones(100, dtype=np.float32) for _ in range(3))
    out = join_audio([[a, b], [c]], 1000, 0.01, 0.05)
    assert out.dtype == np.float32 and out.ndim == 1
    assert len(out) == 300 + 10 + 50
    assert out[100:110].sum() == 0 and out[210:260].sum() == 0
    assert out[:100].min() == 1 and out[260:].min() == 1


def test_join_audio_は1文なら無音を足さない():
    import numpy as np

    from irodori_speaker.webui import join_audio

    a = np.ones(100, dtype=np.float32)
    assert len(join_audio([[a]], 1000, 0.25, 0.6)) == 100


class _FakeSynth:
    def __init__(self, sr=1000):
        self.calls = []
        self.sr = sr

    def generate_once(self, text, **overrides):
        import numpy as np

        self.calls.append((text, overrides))
        return self.sr, np.ones(10, dtype=np.float32), [f"w{len(self.calls)}"]


def _preview_handlers(monkeypatch, tmp_path, fake):
    fns, _ = _handlers(monkeypatch, tmp_path)
    from irodori_speaker import webui

    monkeypatch.setattr(webui.Engine, "is_loaded", lambda self, key: True)
    monkeypatch.setattr(webui.Engine, "load", lambda self, *a, **k: fake)
    return fns


def _run(gen):
    outs = list(gen)
    return outs


def test_generate_は長い文を分けて合成する(monkeypatch, tmp_path):
    fake = _FakeSynth()
    fns = _preview_handlers(monkeypatch, tmp_path, fake)
    outs = _run(fns["generate"](LONG, None, None, None, "narrator", "", 3.0, 3.0, 1.0, "7"))
    assert len(fake.calls) >= 2
    assert "".join(t for t, _ in fake.calls) == LONG
    assert all(len(t) <= 40 for t, _ in fake.calls)
    assert all(o["seed"] == 7 for _, o in fake.calls)
    assert any("(1/" in str(o[1]) for o in outs)
    (sr, audio), status = outs[-1]
    assert sr == 1000 and len(audio) > 10 * len(fake.calls)
    assert f"{len(fake.calls)} 文に分けて合成" in status
    assert "警告" in status


def test_generate_は短い文を1回で合成する(monkeypatch, tmp_path):
    fake = _FakeSynth()
    fns = _preview_handlers(monkeypatch, tmp_path, fake)
    outs = _run(fns["generate"]("こんにちは。", None, None, None, "narrator", "", 3.0, 3.0, 1.0, "7"))
    assert [t for t, _ in fake.calls] == ["こんにちは。"]
    assert "文に分けて" not in outs[-1][1]


def test_generate_はサンプリングレートが違えば失敗する(monkeypatch, tmp_path):
    fake = _FakeSynth()
    orig = fake.generate_once

    def varying(text, **kw):
        fake.sr += 1
        return orig(text, **kw)

    fake.generate_once = varying
    fns = _preview_handlers(monkeypatch, tmp_path, fake)
    with pytest.raises(Exception, match="サンプリングレート"):
        _run(fns["generate"](LONG, None, None, None, "narrator", "", 3.0, 3.0, 1.0, "7"))


def test_compare_emotions_は各文の先頭に絵文字を付ける(monkeypatch, tmp_path):
    fake = _FakeSynth()
    fns = _preview_handlers(monkeypatch, tmp_path, fake)
    from irodori_speaker.config import EMOTIONS

    _run(fns["compare_emotions"](LONG, None, None, None, "narrator", "", 3.0, 3.0, 1.0, "7", 3))
    per = len(fake.calls) // len(EMOTIONS)
    assert per >= 2 and per * len(EMOTIONS) == len(fake.calls)
    texts = [t for t, _ in fake.calls[:per]]
    # 感情ごとに、全ての文の先頭に同じ絵文字（または何も無し）が付く
    prefixes = {t[: min(t.index(c) for c in "あいう" if c in t)] for t in texts}
    assert len(prefixes) == 1
    assert all(t.lstrip("".join(prefixes)).startswith(("あ", "い", "う")) for t in texts)


# -- 分けた文への絵文字の引き継ぎ --


def test_carry_emoji_は先頭の絵文字を後続の文に引き継ぐ():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😠😠あ。", "い。", "う。"]]) == [["😠😠あ。", "😠😠い。", "😠😠う。"]]


def test_carry_emoji_は段落をまたいで引き継ぐ():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😠あ。"], ["い。"]]) == [["😠あ。"], ["😠い。"]]


def test_carry_emoji_は最後に現れた連なりを引き継ぐ():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😠あ。", "い😊う。", "え。"]]) == [["😠あ。", "😠い😊う。", "😊え。"]]


def test_carry_emoji_は文頭の別の絵文字を尊重する():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😠あ。", "😊い。", "う。"]]) == [["😠あ。", "😊い。", "😊う。"]]


def test_carry_emoji_は括弧の直後に付ける():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😠あ。", "「い。」"]]) == [["😠あ。", "「😠い。」"]]


def test_carry_emoji_は括弧の直後の絵文字を文頭の絵文字とみなす():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😠あ。", "「😊い。」", "う。"]]) == [["😠あ。", "「😊い。」", "😊う。"]]


def test_carry_emoji_は絵文字が無ければ入力のまま():
    from irodori_speaker.webui import carry_emoji

    src = [["あ。", "〈い〉。"], ["「う」、…"]]
    assert carry_emoji(src) == src


def test_carry_emoji_は_ZWJ_を含む絵文字を1つの連なりとして引き継ぐ():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["😮‍💨あ。", "い。"]]) == [["😮‍💨あ。", "😮‍💨い。"]]


def test_carry_emoji_は異体字セレクタを絵文字の一部にする():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["⏸️あ。", "い。"]]) == [["⏸️あ。", "⏸️い。"]]


def test_carry_emoji_は絵文字より前の文には付けない():
    from irodori_speaker.webui import carry_emoji

    assert carry_emoji([["あ。", "😠い。", "う。"]]) == [["あ。", "😠い。", "😠う。"]]


def test_generate_は先頭の絵文字を分けた文に引き継ぐ(monkeypatch, tmp_path):
    fake = _FakeSynth()
    fns = _preview_handlers(monkeypatch, tmp_path, fake)
    _run(fns["generate"]("😠😠" + LONG, None, None, None, "narrator", "", 3.0, 3.0, 1.0, "7"))
    assert len(fake.calls) >= 2
    assert all(t.startswith("😠😠") for t, _ in fake.calls)


def test_carry_emoji_は_Irodori_以外の記号を引き継がない():
    from irodori_speaker.webui import carry_emoji

    src = [["室温は20℃です。", "次は5㎏です。", "♪あ。", "★い。"]]
    assert carry_emoji(src) == src
