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
