import json

from irodori_speaker.bgm_scenes import (
    SCENE_SCHEMA,
    SCENE_SYSTEM_PROMPT,
    ScenePlanner,
    chapter_paragraphs,
    chunk_paragraphs,
    load_scenes,
    merge_short,
    save_scenes,
)
from irodori_speaker.bgm_types import Scene
from irodori_speaker.config import LLMConfig


def _scene_reply(*items):
    """items: (start_para, description) → chat の応答。"""
    return {
        "scenes": [
            {
                "start_para": p,
                "description": d,
                "ambient_query": "rain street",
                "mood": "calm",
                "music_query": "calm piano",
            }
            for p, d in items
        ]
    }


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.calls = 0

    def __call__(self, system, user, schema):
        assert system == SCENE_SYSTEM_PROMPT and schema == SCENE_SCHEMA
        self.calls += 1
        self.prompts.append(user)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _paras(n, chars=10):
    return [(i, "あ" * chars) for i in range(n)]


def test_chapter_paragraphs_joins_source_in_seg_order():
    rows = [
        {"chapter": "0", "para": "1", "seg": "1", "source": "B"},
        {"chapter": "0", "para": "0", "seg": "0", "source": "A"},
        {"chapter": "0", "para": "1", "seg": "0", "source": "a"},
        {"chapter": "1", "para": "0", "seg": "0", "source": "C"},
    ]
    assert chapter_paragraphs(rows) == {0: [(0, "A"), (1, "aB")], 1: [(0, "C")]}


def test_chunk_paragraphs_never_splits_a_paragraph():
    paras = [(0, "a" * 2500), (1, "b" * 1000), (2, "c" * 5000)]
    assert [[p for p, _ in c] for c in chunk_paragraphs(paras, 3000)] == [[0], [1], [2]]


def test_chunk_paragraphs_packs_small_paragraphs():
    paras = [(i, "a" * 1000) for i in range(5)]
    assert [[p for p, _ in c] for c in chunk_paragraphs(paras, 3000)] == [[0, 1, 2], [3, 4]]


def test_plan_chapter_carries_previous_scene_across_chunks(tmp_path):
    chat = FakeChat([_scene_reply((0, "rainy street")), _scene_reply()])
    planner = ScenePlanner(LLMConfig(), tmp_path / "scenes.jsonl", chat=chat)
    scenes = planner.plan_chapter(0, _paras(4, 1000), 2000)
    assert [(s.start_para, s.end_para) for s in scenes] == [(0, 3)]
    assert "Previous scene: none (start of chapter)" in chat.prompts[0]
    assert "Previous scene: rainy street" in chat.prompts[1]
    assert "[2] " in chat.prompts[1] and "[0] " not in chat.prompts[1]


def test_plan_chapter_splits_scenes_and_sets_fields(tmp_path):
    chat = FakeChat([_scene_reply((0, "street"), (3, "inn"))])
    planner = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat)
    scenes = planner.plan_chapter(2, _paras(5), 3000)
    assert [(s.chapter, s.index, s.start_para, s.end_para) for s in scenes] == [(2, 0, 0, 2), (2, 1, 3, 4)]
    assert scenes[1].description == "inn" and scenes[1].status == "ok"
    assert scenes[0].ambient_query == "rain street" and scenes[0].music_query == "calm piano"


def test_invalid_reply_retried_once_then_falls_back_to_previous_scene(tmp_path):
    bad = _scene_reply((99, "nowhere"))
    chat = FakeChat([_scene_reply((0, "street")), bad, bad, bad, bad])
    planner = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat)
    scenes = planner.plan_chapter(0, _paras(4, 1000), 2000)
    assert chat.calls == 3  # 1 チャンク目 1 回 + 2 チャンク目 2 回
    assert [(s.start_para, s.end_para) for s in scenes] == [(0, 3)]
    # 失敗はキャッシュしない
    planner2 = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat)
    planner2.plan_chapter(0, _paras(4, 1000), 2000)
    assert chat.calls == 5  # 1 チャンク目はキャッシュ、2 チャンク目は 2 回呼び直し


def test_chat_exception_counts_as_failure(tmp_path):
    chat = FakeChat([RuntimeError("down"), RuntimeError("down")])
    planner = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat)
    scenes = planner.plan_chapter(0, _paras(2), 3000)
    assert scenes[0].status == "llm_error"


def test_first_chunk_failure_gives_silent_llm_error_scene(tmp_path):
    chat = FakeChat([{"scenes": []}, {"bad": 1}])
    planner = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat)
    scenes = planner.plan_chapter(0, _paras(3), 3000)
    assert len(scenes) == 1 and (scenes[0].start_para, scenes[0].end_para) == (0, 2)
    assert scenes[0].status == "llm_error" and scenes[0].ambient_query is None and scenes[0].music_query == ""


def test_first_scene_forced_to_first_paragraph_at_chapter_start(tmp_path):
    chat = FakeChat([_scene_reply((2, "street"), (4, "inn"))])
    planner = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat)
    scenes = planner.plan_chapter(0, _paras(6), 3000)
    assert [(s.start_para, s.end_para) for s in scenes] == [(0, 3), (4, 5)]


def test_validation_rejects_bad_shapes(tmp_path):
    def item(**kw):
        d = {"start_para": 0, "description": "d", "ambient_query": "q", "mood": "m", "music_query": "mq"}
        d.update(kw)
        return {"scenes": [d]}

    bads = [
        item(description=""),
        item(mood="  "),
        item(ambient_query=""),
        item(music_query=None),
        {"scenes": [{"start_para": 1, "description": "d", "ambient_query": None, "mood": "m", "music_query": ""},
                    {"start_para": 1, "description": "d", "ambient_query": None, "mood": "m", "music_query": ""}]},
    ]
    for bad in bads:
        chat = FakeChat([bad, bad])
        planner = ScenePlanner(LLMConfig(), tmp_path / f"{id(bad)}.jsonl", chat=chat)
        assert planner.plan_chapter(0, _paras(3), 3000)[0].status == "llm_error"
    ok = item(ambient_query=None, music_query="")
    chat = FakeChat([ok])
    scenes = ScenePlanner(LLMConfig(), tmp_path / "ok.jsonl", chat=chat).plan_chapter(0, _paras(3), 3000)
    assert scenes[0].status == "ok" and scenes[0].ambient_query is None and scenes[0].music_query == ""


def test_cache_reused(tmp_path):
    path = tmp_path / "s.jsonl"
    chat = FakeChat([_scene_reply((0, "street"))])
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3), 3000)
    a = ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3), 3000)
    assert chat.calls == 1 and a[0].description == "street"


def test_cache_key_depends_on_model_and_text(tmp_path):
    path = tmp_path / "s.jsonl"
    chat = FakeChat([_scene_reply((0, "a")), _scene_reply((0, "b")), _scene_reply((0, "c"))])
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3), 3000)
    ScenePlanner(LLMConfig(model="other"), path, chat=chat).plan_chapter(0, _paras(3), 3000)
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3, 11), 3000)
    assert chat.calls == 3


def test_cache_survives_broken_and_unterminated_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    chat = FakeChat([_scene_reply((0, "a")), _scene_reply((0, "b"))])
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3), 3000)
    with path.open("a", encoding="utf-8") as f:
        f.write("{broken")  # 末尾が改行で終わらない壊れた行
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3), 3000)  # 読み飛ばしてキャッシュ命中
    assert chat.calls == 1
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3, 11), 3000)  # 追記
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[2])


def _sc(start, end, desc="d"):
    return Scene(0, 0, start, end, desc, "q", "m", "mq")


def test_merge_short_merges_into_previous_and_first_into_next():
    chars = {0: 500, 1: 100, 2: 600}
    out = merge_short([_sc(0, 0, "a"), _sc(1, 1, "b"), _sc(2, 2, "c")], chars, 400)
    assert [(s.index, s.start_para, s.end_para, s.description) for s in out] == [(0, 0, 1, "a"), (1, 2, 2, "c")]
    chars = {0: 100, 1: 600}
    out = merge_short([_sc(0, 0, "a"), _sc(1, 1, "b")], chars, 400)
    assert [(s.start_para, s.end_para, s.description) for s in out] == [(0, 1, "b")]


def test_merge_short_whole_chapter_short_gives_one_scene():
    out = merge_short([_sc(0, 0), _sc(1, 1), _sc(2, 2)], {0: 100, 1: 100, 2: 100}, 400)
    assert [(s.start_para, s.end_para) for s in out] == [(0, 2)]
    assert merge_short([_sc(0, 1)], {0: 10, 1: 10}, 400)[0].index == 0


def test_save_scenes_keeps_other_chapters(tmp_path):
    path = tmp_path / "scenes.json"
    s0 = Scene(0, 0, 0, 3, "zero", None, "m", "")
    s1 = Scene(1, 0, 0, 3, "one", "q", "m", "mq")
    s2 = Scene(2, 0, 0, 3, "two", "q", "m", "mq")
    save_scenes(path, {0: [s0], 2: [s2]}, {0, 1, 2})
    save_scenes(path, {1: [s1]}, {0, 1})
    loaded = load_scenes(path)
    assert loaded == {0: [s0], 1: [s1]}
    assert load_scenes(tmp_path / "none.json") == {}


def test_empty_reply_at_chapter_start_is_invalid_and_not_cached(tmp_path):
    path = tmp_path / "s.jsonl"
    chat = FakeChat([{"scenes": []}, _scene_reply((0, "street"))])
    scenes = ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(3), 3000)
    assert chat.calls == 2 and scenes[0].status == "ok" and scenes[0].description == "street"  # 1 回だけやり直す
    chat = FakeChat([{"scenes": []}, {"scenes": []}, _scene_reply((0, "inn"))])
    path2 = tmp_path / "s2.jsonl"
    assert ScenePlanner(LLMConfig(), path2, chat=chat).plan_chapter(0, _paras(3), 3000)[0].status == "llm_error"
    assert not path2.exists() or path2.read_text(encoding="utf-8") == ""  # キャッシュしない
    # 再実行では問い合わせ直す
    assert ScenePlanner(LLMConfig(), path2, chat=chat).plan_chapter(0, _paras(3), 3000)[0].description == "inn"


def test_empty_reply_in_later_chunk_is_valid_and_cached(tmp_path):
    path = tmp_path / "s.jsonl"
    chat = FakeChat([_scene_reply((0, "street")), {"scenes": []}])
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(4, 1000), 2000)
    ScenePlanner(LLMConfig(), path, chat=chat).plan_chapter(0, _paras(4, 1000), 2000)
    assert chat.calls == 2


def test_merge_short_handles_chapter_start_before_merging_forward():
    chars = {0: 200, 1: 200, 2: 600}
    out = merge_short([_sc(0, 0, "a"), _sc(1, 1, "b"), _sc(2, 2, "c")], chars, 400)
    assert [(s.index, s.start_para, s.end_para, s.description) for s in out] == [(0, 0, 1, "b"), (1, 2, 2, "c")]


def test_missing_ambient_query_key_is_invalid(tmp_path):
    reply = _scene_reply((0, "street"))
    del reply["scenes"][0]["ambient_query"]
    chat = FakeChat([reply, reply])
    scenes = ScenePlanner(LLMConfig(), tmp_path / "s.jsonl", chat=chat).plan_chapter(0, _paras(3), 3000)
    assert chat.calls == 2 and scenes[0].status == "llm_error"
