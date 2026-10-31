import pytest

pytest.importorskip("gradio")

from irodori_speaker import webui  # noqa: E402


def test_rows_from_dataframe_blanks_break_text():
    base = {"chapter": 0, "para": 0, "seg": 0, "labels": "", "source": "＊"}
    original = [{**base, "kind": "break", "text": ""}, {**base, "para": 1, "kind": "narration", "text": "あ", "source": "あ"}]
    df = [[0, 0, 0, "break", "", "読まれる？", "＊"], [0, 1, 0, "narration", "", "あ！", "あ"]]
    got = webui.rows_from_dataframe(df, original)
    assert got[0]["text"] == "" and got[1]["text"] == "あ！"
