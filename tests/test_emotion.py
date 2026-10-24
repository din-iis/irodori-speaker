import copy
import json

import pytest
import requests

from irodori_speaker import emotion
from irodori_speaker.config import EMOTIONS, LLMConfig
from irodori_speaker.emotion import (
    JSON_SCHEMA,
    Annotation,
    LLMAnnotator,
    LLMServer,
    apply_shift,
    build_spoken_text,
    build_user_prompt,
    labels_from_str,
    labels_to_str,
    list_models,
    paragraph_hash,
    split_clauses,
    to_emoji,
    validate,
)
from irodori_speaker.text_splitter import Segment


def segs(*texts):
    return [Segment(0, 0, i, "dialogue", t) for i, t in enumerate(texts)]


# ---- split_clauses ----


def test_split_clauses_examples():
    assert split_clauses("「やあ。元気？」と彼は言った。") == ["「やあ。", "元気？」", "と彼は言った。"]
    assert split_clauses("あははっ、それ本当に言ってるの？…まぁ、君らしいけどね。") == [
        "あははっ、",
        "それ本当に言ってるの？…",
        "まぁ、",
        "君らしいけどね。",
    ]


def test_split_clauses_groups_consecutive_punctuation():
    assert split_clauses("まさか！！本当？！") == ["まさか！！", "本当？！"]
    assert split_clauses("そう…。うん") == ["そう…。", "うん"]


def test_split_clauses_attaches_closing_brackets():
    assert split_clauses("「元気？」") == ["「元気？」"]
    assert split_clauses("（はい。）そうです") == ["（はい。）", "そうです"]


def test_split_clauses_edge_cases():
    assert split_clauses("") == []
    assert split_clauses("句読点なし") == ["句読点なし"]
    assert split_clauses("  先頭空白、あり") == ["  先頭空白、", "あり"]


@pytest.mark.parametrize(
    "s",
    ["「やあ。元気？」と彼は言った。", "あ、い。 う！ ", "。。。", "「」", "、「", "abc, def!? ghi", " 　"],
)
def test_split_clauses_roundtrip(s):
    assert "".join(split_clauses(s)) == s


def test_split_clauses_merges_bracket_or_space_only_fragments():
    # 括弧や空白だけの断片は前の句に併合する
    assert split_clauses("あ。 「") == ["あ。 「"]
    assert split_clauses("あ。 ") == ["あ。 "]


# ---- prompt / schema ----


def test_json_schema_enum_and_required():
    item = JSON_SCHEMA["properties"]["segments"]["items"]
    assert item["properties"]["emotion"]["enum"] == EMOTIONS
    assert item["required"] == ["s", "c", "emotion", "intensity"]


def test_build_user_prompt():
    p = build_user_prompt("前", [["あ、", "い。"], ["う。"]], "次")
    assert p == "【直前の段落（文脈）】\n前\n【対象の句】\n0-0: あ、\n0-1: い。\n1-0: う。\n【直後の段落（文脈）】\n次"
    assert build_user_prompt(None, [["あ。"]], None) == "【対象の句】\n0-0: あ。"


# ---- validate ----


def test_validate_shape_and_fallbacks():
    data = {
        "segments": [
            {"s": 0, "c": 0, "emotion": "happy", "intensity": 3},
            {"s": 0, "c": 1, "emotion": "bogus", "intensity": 2},
            {"s": 1, "c": 0, "emotion": "sad", "intensity": 2},
        ]
    }
    anns = validate(data, [2, 1, 2])
    assert [len(a) for a in anns] == [2, 1, 2]
    assert anns[0][0] == Annotation("happy", 3)
    assert anns[0][1] == Annotation("neutral", 1)  # 不正な感情
    assert anns[1][0] == Annotation("sad", 2)
    assert anns[2] == [Annotation(), Annotation()]  # 欠落


def test_validate_clamps_and_rejects_garbage():
    one = lambda **kw: validate({"segments": [{"s": 0, "c": 0, "emotion": "sad", **kw}]}, [1])[0][0]
    assert one(intensity=9).intensity == 3
    assert one(intensity=0).intensity == 1
    assert one(intensity="x").intensity == 2
    assert one(intensity=True).intensity == 2
    assert validate("oops", [1, 1]) == [[Annotation()], [Annotation()]]
    assert validate({"segments": "x"}, [1]) == [[Annotation()]]


def test_validate_out_of_range_and_bool_indices():
    for item in (
        {"s": 5, "c": 0},
        {"s": 0, "c": 5},
        {"s": -1, "c": 0},
        {"s": True, "c": 0},
        {"s": 0, "c": True},
        {"s": "0", "c": 0},
        "str",
    ):
        data = {"segments": [item if isinstance(item, str) else {**item, "emotion": "sad", "intensity": 2}]}
        assert validate(data, [1]) == [[Annotation()]]


# ---- apply_shift ----


def test_apply_shift():
    assert apply_shift(Annotation("happy", 1), "narration", 1) == Annotation("neutral", 1)
    assert apply_shift(Annotation("happy", 3), "narration", 1) == Annotation("happy", 2)
    assert apply_shift(Annotation("happy", 1), "dialogue", 1) == Annotation("happy", 1)


# ---- to_emoji / build_spoken_text ----


def test_to_emoji_default_and_override():
    assert to_emoji(Annotation("happy", 2)) == "😊"
    assert to_emoji(Annotation("neutral", 3)) == ""
    assert to_emoji(Annotation("happy", 2), {"happy": ["a", "b", "c"]}) == "b"
    assert to_emoji(Annotation("angry", 3)) == "😠😠"


def test_build_spoken_text_inserts_at_head_and_after_bracket():
    anns = [Annotation("happy", 2)] * 2
    assert build_spoken_text(["「やあ。", "元気？」"], anns) == "「😊やあ。元気？」"
    assert build_spoken_text(["「「やあ。"], anns[:1]) == "「😊「やあ。"
    assert build_spoken_text(["やあ。"], anns[:1]) == "😊やあ。"


def test_build_spoken_text_suppresses_repeats():
    t = {"happy": ["", "😊", "😆"]}
    anns = [Annotation("happy", 2), Annotation("happy", 2), Annotation("neutral", 1), Annotation("happy", 2)]
    # [😊, 😊, "", 😊] → 1 句目だけ、3 句目は空で挿入なし、4 句目は直前(空)と違うので再挿入
    assert build_spoken_text(["あ、", "い、", "う、", "え。"], anns, t) == "😊あ、い、う、😊え。"


def test_build_spoken_text_neutral_unchanged_and_short_anns():
    clauses = ["あ、", "い。"]
    assert build_spoken_text(clauses, [Annotation()] * 2) == "あ、い。"
    # anns が短ければ不足分は neutral
    assert build_spoken_text(clauses, [Annotation("happy", 2)]) == "😊あ、い。"


def test_build_spoken_text_table_override():
    anns = [Annotation("sad", 3)]
    assert build_spoken_text(["あ。"], anns, {"sad": ["", "", "T"]}) == "Tあ。"


# ---- labels ----


def test_labels_roundtrip():
    anns = [Annotation("happy", 2), Annotation("neutral", 1)]
    assert labels_to_str(anns) == "happy:2|neutral:1"
    assert labels_from_str("happy:2|neutral:1") == anns
    assert labels_from_str("") == []
    assert labels_to_str([]) == ""


# ---- paragraph_hash ----


def test_paragraph_hash_depends_on_model_and_clauses():
    h = paragraph_hash("m", [["あ、", "い。"]])
    assert h == paragraph_hash("m", [["あ、", "い。"]])
    assert h != paragraph_hash("m2", [["あ、", "い。"]])
    assert h != paragraph_hash("m", [["あ、い。"]])


# ---- LLMAnnotator ----


def test_annotator_caches_and_survives_failure(tmp_path, monkeypatch):
    cache = tmp_path / "ann.jsonl"
    calls = []
    ann = LLMAnnotator(LLMConfig(), cache)

    def fake_chat(prompt):
        calls.append(prompt)
        return {"segments": [{"s": 0, "c": 0, "emotion": "angry", "intensity": 2}]}

    monkeypatch.setattr(ann, "_chat", fake_chat)
    target = segs("ふざけるな！")
    assert ann.annotate(target)[0][0] == Annotation("angry", 2)
    assert ann.annotate(target)[0][0] == Annotation("angry", 2)
    assert len(calls) == 1
    row = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert row["annotations"] == [[{"emotion": "angry", "intensity": 2}]]

    ann2 = LLMAnnotator(LLMConfig(), cache)
    monkeypatch.setattr(ann2, "_chat", lambda p: (_ for _ in ()).throw(AssertionError("再問い合わせ")))
    assert ann2.annotate(target)[0][0] == Annotation("angry", 2)

    ann3 = LLMAnnotator(LLMConfig(), tmp_path / "other.jsonl")
    monkeypatch.setattr(ann3, "_chat", lambda p: (_ for _ in ()).throw(requests.ConnectionError("x")))
    assert ann3.annotate(segs("こんにちは、どうも"))[0] == [Annotation(), Annotation()]
    assert ann3.cache == {}
    assert not (tmp_path / "other.jsonl").exists()


def test_annotator_prompt_uses_context(tmp_path, monkeypatch):
    ann = LLMAnnotator(LLMConfig(), tmp_path / "a.jsonl")
    seen = []
    monkeypatch.setattr(ann, "_chat", lambda p: seen.append(p) or {"segments": []})
    ann.annotate(segs("あ。"), prev=segs("前1。", "前2。"), nxt=segs("次。"))
    assert "前1。前2。" in seen[0] and "次。" in seen[0] and "0-0: あ。" in seen[0]


def test_annotate_many_keeps_order(tmp_path, monkeypatch):
    ann = LLMAnnotator(LLMConfig(concurrency=2), tmp_path / "a.jsonl")
    emos = {"A。": "happy", "B。": "sad", "C。": "angry"}

    def fake_chat(prompt):
        import time

        time.sleep(0.05 if "A。" in prompt else 0)
        text = next(k for k in emos if k in prompt)
        return {"segments": [{"s": 0, "c": 0, "emotion": emos[text], "intensity": 2}]}

    monkeypatch.setattr(ann, "_chat", fake_chat)
    items = [(segs(t), None, None) for t in ("A。", "B。", "C。")]
    out = ann.annotate_many(items)
    assert [o[0][0].emotion for o in out] == ["happy", "sad", "angry"]


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


def test_chat_ollama_payload(tmp_path, monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured.update(kw)
        return FakeResp({"message": {"content": '{"segments": []}'}})

    monkeypatch.setattr(requests, "post", fake_post)
    ann = LLMAnnotator(LLMConfig(base_url="http://h:1/"), tmp_path / "a.jsonl")
    assert ann._chat("u") == {"segments": []}
    assert captured["url"] == "http://h:1/api/chat"
    body = captured["json"]
    assert body["think"] is False
    assert body["format"] == JSON_SCHEMA
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0}
    assert body["messages"][1] == {"role": "user", "content": "u"}
    assert captured["timeout"] == 120.0


def test_chat_openai_falls_back_on_400_and_strips_think(tmp_path, monkeypatch):
    bodies = []

    def fake_post(url, **kw):
        assert url == "http://h:1/v1/chat/completions"
        assert kw["headers"]["Authorization"] == "Bearer k"
        bodies.append(copy.deepcopy(kw["json"]))
        if len(bodies) < 3:
            return FakeResp(status=400)
        return FakeResp({"choices": [{"message": {"content": '<think>\nうーん\n</think>\n{"segments": []}'}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    cfg = LLMConfig(api="openai", base_url="http://h:1/v1", api_key="k")
    ann = LLMAnnotator(cfg, tmp_path / "a.jsonl")
    assert ann._chat("u") == {"segments": []}
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[0]["reasoning_effort"] == "none"
    assert bodies[1]["response_format"] == {"type": "json_object"}
    assert "response_format" not in bodies[2]
    assert "chat_template_kwargs" not in bodies[2]
    assert "reasoning_effort" not in bodies[2]


def test_chat_openai_think_true_omits_reasoning_effort(tmp_path, monkeypatch):
    bodies = []

    def fake_post(url, **kw):
        bodies.append(kw["json"])
        assert "Authorization" not in kw["headers"]
        return FakeResp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(requests, "post", fake_post)
    ann = LLMAnnotator(LLMConfig(api="openai", base_url="http://h/v1", think=True), tmp_path / "a.jsonl")
    ann._chat("u")
    assert "reasoning_effort" not in bodies[0]
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_unknown_api_raises(tmp_path):
    ann = LLMAnnotator(LLMConfig(api="bogus"), tmp_path / "a.jsonl")
    with pytest.raises(ValueError):
        ann._chat("u")


def test_unload(tmp_path, monkeypatch):
    posts = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: posts.append((url, kw["json"])) or FakeResp({}))
    LLMAnnotator(LLMConfig(base_url="http://h:1"), tmp_path / "a.jsonl").unload()
    assert posts == [("http://h:1/api/generate", {"model": "qwen3.5:4b-q8_0", "keep_alive": 0})]

    posts.clear()
    LLMAnnotator(LLMConfig(api="openai", base_url="http://h/v1"), tmp_path / "b.jsonl").unload()
    LLMAnnotator(LLMConfig(unload=False), tmp_path / "c.jsonl").unload()
    assert posts == []


def test_unload_failure_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    LLMAnnotator(LLMConfig(), tmp_path / "a.jsonl").unload()
    assert "[emotion]" in capsys.readouterr().out


# ---- LLMServer ----


class FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def test_server_not_started_when_healthy(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(status=200))
    popen = []
    monkeypatch.setattr(emotion.subprocess, "Popen", lambda *a, **k: popen.append(a) or FakeProc())
    with LLMServer(LLMConfig(server_command=["srv"])):
        pass
    assert popen == []


def test_server_noop_without_command(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("呼ばない")))
    with LLMServer(LLMConfig()):
        pass


def test_server_starts_and_terminates(monkeypatch):
    proc = FakeProc()
    popen = []
    monkeypatch.setattr(emotion.subprocess, "Popen", lambda *a, **k: popen.append(a[0]) or proc)
    monkeypatch.setattr(emotion.time, "sleep", lambda s: None)
    state = {"n": 0}

    def fake_get(url, **kw):
        state["n"] += 1
        if state["n"] < 3:
            raise requests.ConnectionError("x")
        return FakeResp(status=200)

    monkeypatch.setattr(requests, "get", fake_get)
    with LLMServer(LLMConfig(server_command=["srv", "--x"])):
        assert popen == [["srv", "--x"]]
        assert not proc.terminated
    assert proc.terminated and not proc.killed


def test_server_timeout_raises_and_terminates(monkeypatch):
    proc = FakeProc()
    monkeypatch.setattr(emotion.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(emotion.time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    with pytest.raises(RuntimeError):
        with LLMServer(LLMConfig(server_command=["srv"], server_start_timeout=3)):
            pass
    assert proc.terminated


# ---- list_models ----


def test_list_models(monkeypatch):
    urls = []

    def fake_get(url, **kw):
        urls.append(url)
        if url.endswith("/api/tags"):
            return FakeResp({"models": [{"name": "a:1"}, {"name": "b:2"}]})
        return FakeResp({"data": [{"id": "x"}, {"id": "y"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    assert list_models(LLMConfig(base_url="http://h:1/")) == ["a:1", "b:2"]
    assert list_models(LLMConfig(api="openai", base_url="http://h/v1")) == ["x", "y"]
    assert urls == ["http://h:1/api/tags", "http://h/v1/models"]


def test_list_models_failure_propagates(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
    with pytest.raises(requests.ConnectionError):
        list_models(LLMConfig())
