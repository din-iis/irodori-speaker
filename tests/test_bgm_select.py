import json
from pathlib import Path

import pytest

from irodori_speaker.bgm_select import (
    GATE_PROMPT_VERSION,
    Gate,
    Slot,
    build_gate_prompt,
    build_slots,
    choose,
    gate_slots,
    keyword_reject,
    load_selection,
    read_exclude,
    search_slots,
    window_starts,
    write_gate_tsv,
    write_missing,
    write_selection,
)
from irodori_speaker.bgm_types import Choice, Scene, Sound
from irodori_speaker.config import BgmConfig, LLMConfig
from irodori_speaker.freesound import DailyLimit, FreesoundError


def snd(name="s", id=1, tags=None, description="", duration=60.0):
    return Sound(
        id=id,
        name=name,
        tags=tags or [],
        description=description,
        duration=duration,
        license="http://creativecommons.org/licenses/by/4.0/",
        username="u",
        url=f"https://freesound.org/s/{id}/",
        preview=f"https://cdn/{id}.mp3",
    )


def scene(index, ambient="rain", mood="calm", music="soft piano", chapter=0):
    return Scene(
        chapter=chapter,
        index=index,
        start_para=index * 10,
        end_para=index * 10 + 9,
        description=f"scene {index}",
        ambient_query=ambient,
        mood=mood,
        music_query=music,
    )


def slot(kind="ambient", chapter=0, sc=0, query="rain", duration="[30 TO 900]"):
    return Slot(chapter, sc, kind, query, duration, "the sound of rain", "rain on a roof")


class FakeClient:
    def __init__(self, results=None, errors=None):
        self.results = results or {}
        self.errors = errors or {}
        self.calls = []

    def search(self, query, duration, n):
        self.calls.append((query, duration, n))
        if query in self.errors:
            raise self.errors[query]
        return list(self.results.get(query, []))


# ---------- build_slots ----------


def test_build_slots():
    scenes = {
        0: [
            scene(0, ambient="rain", music="soft piano"),
            scene(1, ambient=None, music="dark drone"),
            scene(2, ambient="wind", music=""),
        ]
    }
    slots = build_slots(scenes, BgmConfig())
    assert [s.key for s in slots] == ["0:0:ambient", "0:0:intro", "0:1:bridge", "0:2:ambient"]
    amb, intro, bridge, _ = slots
    assert amb.query == "rain"
    assert amb.clap_text == "the sound of scene 0"
    assert amb.gate_text == "scene 0"
    assert amb.duration == BgmConfig().ambient_duration
    assert intro.query == "soft piano"
    assert intro.clap_text == "a calm piece of music"
    assert intro.gate_text == "mood: calm"
    assert intro.duration == BgmConfig().music_duration
    assert bridge.query == "dark drone"


# ---------- search_slots ----------


def test_search_uses_cache_on_rerun(tmp_path):
    cache = tmp_path / "search.json"
    slots = [slot(query="rain"), slot(sc=1, query="wind")]
    client = FakeClient({"rain": [snd(id=1)], "wind": [snd(id=2)]})
    found, failures = search_slots(slots, client, cache, 12)
    assert failures == {}
    assert [s.id for s in found["0:0:ambient"]] == [1]
    assert len(client.calls) == 2
    found2, _ = search_slots(slots, client, cache, 12)
    assert len(client.calls) == 2  # 2 回目は API を呼ばない
    assert [s.id for s in found2["0:1:ambient"]] == [2]


def test_search_error_marks_slot_and_is_not_cached(tmp_path):
    cache = tmp_path / "search.json"
    slots = [slot(query="rain"), slot(sc=1, query="wind")]
    client = FakeClient({"wind": [snd(id=2)]}, errors={"rain": FreesoundError("x")})
    found, failures = search_slots(slots, client, cache, 12)
    assert failures == {"0:0:ambient": "error"}
    assert "0:0:ambient" not in found
    assert [s.id for s in found["0:1:ambient"]] == [2]
    client2 = FakeClient({"rain": [snd(id=1)]})
    found2, failures2 = search_slots(slots, client2, cache, 12)
    assert failures2 == {}
    assert [c[0] for c in client2.calls] == ["rain"]  # 失敗した分だけやり直す


def test_daily_limit_propagates_and_keeps_earlier_results_cached(tmp_path):
    cache = tmp_path / "search.json"
    slots = [slot(query="rain"), slot(sc=1, query="wind")]
    client = FakeClient({"rain": [snd(id=1)]}, errors={"wind": DailyLimit("limit")})
    with pytest.raises(DailyLimit):
        search_slots(slots, client, cache, 12)
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert list(data) == ["rain|[30 TO 900]|12"]
    client2 = FakeClient({"wind": [snd(id=2)]})
    found, _ = search_slots(slots, client2, cache, 12)
    assert [c[0] for c in client2.calls] == ["wind"]
    assert [s.id for s in found["0:0:ambient"]] == [1]


# ---------- keyword_reject ----------


def test_keyword_reject():
    assert keyword_reject(snd("Japan_Matsudo_Station_Train_to_Kita-Kashiwa_Annoucements_Engines")) == "annouc"
    assert keyword_reject(snd("Spacious Bar Ambience With Music, Medium Crowd 01")) == "music"
    assert keyword_reject(snd("Ambience_Bar_No_Music_3.wav")) is None
    assert keyword_reject(snd("bar chatter", tags=["music", "announcement"])) is None  # タグは見ない


# ---------- Gate ----------


class FakeChat:
    """accept(sound id) で採否を決める偽の chat。プロンプトを記録する。"""

    def __init__(self, accept=lambda i: True, responses=None):
        self.accept = accept
        self.responses = list(responses) if responses else None
        self.calls = []

    def __call__(self, cfg, system, user, schema, name):
        self.calls.append({"system": system, "user": user})
        if self.responses is not None:
            r = self.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        ids = [int(line.split("|")[0].strip()[3:]) for line in user.split("Candidates:\n")[1].splitlines()]
        return {"results": [{"id": i, "accept": self.accept(i), "reason": f"r{i}"} for i in ids]}


def make_gate(tmp_path, chat):
    return Gate(LLMConfig(), tmp_path / "gate.jsonl", chat=chat)


def test_keyword_rejected_ambient_not_sent_to_llm(tmp_path):
    chat = FakeChat()
    gate = make_gate(tmp_path, chat)
    s = slot("ambient")
    good, bad = snd("Quiet rain", id=1), snd("Station Announcements", id=2)
    survivors, rows, llm_error = gate_slots([s], {s.key: [good, bad]}, gate, 1)
    assert [x.id for x in survivors[s.key]] == [1]
    assert llm_error == {}
    assert "Station Announcements" not in chat.calls[0]["user"]
    by_id = {r["id"]: r for r in rows}
    assert by_id[2]["accept"] is False and by_id[2]["reason"] == "keyword: announce"
    assert by_id[1]["accept"] is True


def test_keyword_reject_not_used_for_music(tmp_path):
    chat = FakeChat()
    gate = make_gate(tmp_path, chat)
    s = slot("intro", query="piano")
    sound = snd("Calm Music Piece", id=5)
    survivors, rows, _ = gate_slots([s], {s.key: [sound]}, gate, 1)
    assert [x.id for x in survivors[s.key]] == [5]
    assert "Calm Music Piece" in chat.calls[0]["user"]
    assert chat.calls[0]["system"].startswith("You choose short instrumental music")


def test_all_keyword_rejected_skips_llm(tmp_path):
    chat = FakeChat()
    gate = make_gate(tmp_path, chat)
    s = slot("ambient")
    survivors, rows, llm_error = gate_slots([s], {s.key: [snd("radio chatter", id=3)]}, gate, 1)
    assert survivors[s.key] == []
    assert chat.calls == []
    assert llm_error == {}
    assert rows[0]["reason"] == "keyword: radio"


def test_gate_prompt_is_one_line_per_candidate():
    sound = snd("na\nme\t x", id=7, tags=["a", "b"], description="line1\n\tline2 " + "x" * 5000)
    prompt = build_gate_prompt("rain on a roof", [sound, snd("other", id=8)])
    head, body = prompt.split("Candidates:\n")
    assert head == "Scene: rain on a roof\n\n"
    lines = body.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("id=7 | name=na me x | tags=a b | description=line1 line2 ")
    desc = lines[0].split("description=")[1]
    assert len(desc) <= 300
    assert "\t" not in lines[0]


def test_gate_rejects_missing_ids_as_failure(tmp_path):
    bad = {"results": [{"id": 1, "accept": True, "reason": "ok"}]}  # id 2 が欠ける
    chat = FakeChat(responses=[bad, bad])
    gate = make_gate(tmp_path, chat)
    assert gate.judge("ambient", "rain", [snd(id=1), snd(id=2)]) is None
    assert len(chat.calls) == 2  # 1 回やり直す
    assert not (tmp_path / "gate.jsonl").exists() or (tmp_path / "gate.jsonl").read_text() == ""
    # 失敗はキャッシュされないので、次は正しい応答で通る
    gate2 = make_gate(tmp_path, FakeChat())
    assert gate2.judge("ambient", "rain", [snd(id=1), snd(id=2)]) is not None


def test_gate_retry_succeeds_on_second_try(tmp_path):
    bad = {"results": []}
    good = {"results": [{"id": 1, "accept": False, "reason": "no"}]}
    gate = make_gate(tmp_path, FakeChat(responses=[bad, good]))
    assert gate.judge("ambient", "rain", [snd(id=1)]) == {1: (False, "no")}


def test_gate_rejects_wrong_types_and_duplicate_ids(tmp_path):
    dup = {"results": [{"id": 1, "accept": True, "reason": "a"}, {"id": 1, "accept": True, "reason": "b"}]}
    typed = {"results": [{"id": 1, "accept": "yes", "reason": "a"}]}
    assert make_gate(tmp_path, FakeChat(responses=[dup, dup])).judge("ambient", "x", [snd(id=1)]) is None
    assert make_gate(tmp_path, FakeChat(responses=[typed, typed])).judge("ambient", "x", [snd(id=1)]) is None


def test_gate_chat_exception_is_failure(tmp_path):
    import requests

    chat = FakeChat(responses=[requests.ConnectionError("down"), requests.ConnectionError("down")])
    assert make_gate(tmp_path, chat).judge("ambient", "x", [snd(id=1)]) is None


def test_gate_caches_per_candidate_and_asks_only_new(tmp_path):
    chat = FakeChat()
    gate = make_gate(tmp_path, chat)
    gate.judge("ambient", "rain", [snd(id=1)])
    assert len(chat.calls) == 1
    # 新しい Gate（jsonl から読み直す）。id=1 は問い合わせず、id=2 だけ聞く
    chat2 = FakeChat()
    gate2 = make_gate(tmp_path, chat2)
    res = gate2.judge("ambient", "rain", [snd(id=1), snd(id=2)])
    assert set(res) == {1, 2}
    assert len(chat2.calls) == 1
    assert "id=1 " not in chat2.calls[0]["user"] and "id=2 " in chat2.calls[0]["user"]
    # すべてキャッシュ済みなら呼ばない
    chat3 = FakeChat()
    make_gate(tmp_path, chat3).judge("ambient", "rain", [snd(id=1), snd(id=2)])
    assert chat3.calls == []


def test_gate_cache_key_differs_by_kind_and_text(tmp_path):
    chat = FakeChat()
    gate = make_gate(tmp_path, chat)
    gate.judge("ambient", "rain", [snd(id=1)])
    gate.judge("intro", "rain", [snd(id=1)])
    gate.judge("ambient", "wind", [snd(id=1)])
    assert len(chat.calls) == 3
    assert isinstance(GATE_PROMPT_VERSION, int)


def test_gate_slots_writes_rows_and_llm_error(tmp_path):
    ok, ng = slot(sc=0, query="rain"), slot(sc=1, query="wind")
    found = {ok.key: [snd("rain a", id=1), snd("rain b", id=2)], ng.key: [snd("wind a", id=3)]}

    def chat(cfg, system, user, schema, name):
        if "wind a" in user:
            return {"results": []}  # 壊れた応答（2 回とも）
        return {"results": [
            {"id": 1, "accept": True, "reason": "fits"},
            {"id": 2, "accept": False, "reason": "birds"},
        ]}

    survivors, rows, llm_error = gate_slots([ok, ng], found, Gate(LLMConfig(), tmp_path / "g.jsonl", chat=chat), 2)
    assert [s.id for s in survivors[ok.key]] == [1]
    assert survivors[ng.key] == []
    assert llm_error == {ng.key: "llm_error"}
    assert [(r["chapter"], r["scene"], r["id"], r["accept"], r["reason"]) for r in rows] == [
        (0, 0, 1, True, "fits"),
        (0, 0, 2, False, "birds"),
    ]
    tsv = tmp_path / "gate.tsv"
    write_gate_tsv(tsv, rows)
    lines = tsv.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t")[:2] == ["chapter", "scene"]
    assert len(lines) == 3
    assert all(len(line.split("\t")) == len(lines[0].split("\t")) for line in lines)


def test_gate_slots_skips_slots_without_candidates(tmp_path):
    chat = FakeChat()
    s = slot()
    survivors, rows, llm_error = gate_slots([s], {s.key: []}, make_gate(tmp_path, chat), 1)
    assert survivors == {s.key: []} and rows == [] and llm_error == {}
    assert chat.calls == []
    # 検索失敗（found に無い）の slot も呼ばない
    survivors, _, _ = gate_slots([s], {}, make_gate(tmp_path, chat), 1)
    assert chat.calls == [] and survivors.get(s.key, []) == []


def test_write_gate_tsv_sanitizes_tabs_and_newlines(tmp_path):
    rows = [{"chapter": 0, "scene": 0, "kind": "ambient", "id": 1, "name": "a\tb\nc", "accept": True, "reason": "x\ty"}]
    path = tmp_path / "gate.tsv"
    write_gate_tsv(path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert len(lines[1].split("\t")) == len(lines[0].split("\t"))
    assert lines[1].split("\t")[0] == "1"  # 人が読む表なので章は 1 始まり


# ---------- window_starts ----------


def test_window_starts():
    assert window_starts(8.0) == [0.0]
    assert window_starts(10.0) == [0.0]
    assert window_starts(30.0) == [0.0, 10.0, 20.0]
    assert window_starts(600.0) == pytest.approx([0, 118, 236, 354, 472, 590])


# ---------- choose ----------


class FakeScorer:
    model = "fake-clap"

    def __init__(self, points):
        self.points = points  # sound id -> score
        self.calls = []

    def score(self, path, duration, text):
        self.calls.append((Path(path).name, duration, text))
        assert Path(path).exists()
        return self.points[int(Path(path).stem)]


class FakeDownload:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def __call__(self, url, dst):
        sid = int(url.rsplit("/", 1)[1].split(".")[0])
        self.calls.append(sid)
        if sid in self.fail:
            raise FreesoundError("x")
        Path(dst).write_bytes(b"x")


def run_choose(tmp_path, slots, found, survivors, failures=None, points=None, exclude=None, download=None):
    scorer = FakeScorer(points or {})
    download = download or FakeDownload()
    choices = choose(
        slots,
        found,
        survivors,
        failures or {},
        scorer,
        tmp_path / "scores.json",
        exclude or set(),
        tmp_path / "dl",
        2,
        download=download,
    )
    return choices, scorer, download


def test_choose_picks_highest_score_among_survivors(tmp_path):
    s = slot()
    sounds = [snd(id=1), snd(id=2), snd(id=3)]
    choices, scorer, _ = run_choose(
        tmp_path, [s], {s.key: sounds}, {s.key: sounds[:2]}, points={1: 0.2, 2: 0.5, 3: 0.9}
    )
    (c,) = choices
    assert c.sound.id == 2 and c.score == pytest.approx(0.5) and c.reason is None
    assert (c.chapter, c.scene, c.kind, c.query) == (0, 0, "ambient", "rain")
    assert {name for name, _, _ in scorer.calls} == {"1.mp3", "2.mp3"}  # 3 は残っていない
    assert all(text == "the sound of rain" for _, _, text in scorer.calls)
    assert not list((tmp_path / "dl").glob("*"))  # 採点後に消す


def test_choose_excluded_id_falls_to_next_then_all_rejected(tmp_path):
    s = slot()
    sounds = [snd(id=1), snd(id=2)]
    pts = {1: 0.2, 2: 0.5}
    choices, _, _ = run_choose(tmp_path, [s], {s.key: sounds}, {s.key: sounds}, points=pts, exclude={2})
    assert choices[0].sound.id == 1
    choices, _, _ = run_choose(tmp_path, [s], {s.key: sounds}, {s.key: sounds}, points=pts, exclude={1, 2})
    assert choices[0].sound is None and choices[0].reason == "all_rejected"
    assert choices[0].score is None


def test_choose_reasons_no_result_all_rejected_error(tmp_path):
    a, b, c, d, e = (slot(sc=i) for i in range(5))
    found = {a.key: [], b.key: [snd(id=1)], d.key: [snd(id=2)], e.key: [snd(id=3)]}
    survivors = {b.key: [], d.key: [], e.key: []}
    failures = {c.key: "error", d.key: "llm_error"}
    choices, _, _ = run_choose(tmp_path, [a, b, c, d, e], found, survivors, failures)
    assert [ch.reason for ch in choices] == ["no_result", "all_rejected", "error", "llm_error", "all_rejected"]
    assert all(ch.sound is None for ch in choices)


def test_choose_error_takes_precedence_over_everything(tmp_path):
    s = slot()
    choices, _, _ = run_choose(tmp_path, [s], {}, {}, {s.key: "error"})
    assert choices[0].reason == "error"


def test_choose_uses_score_cache_without_download(tmp_path):
    s = slot()
    sounds = [snd(id=1), snd(id=2)]
    run_choose(tmp_path, [s], {s.key: sounds}, {s.key: sounds}, points={1: 0.2, 2: 0.5})
    choices, scorer, download = run_choose(
        tmp_path, [s], {s.key: sounds}, {s.key: sounds}, points={}
    )
    assert choices[0].sound.id == 2
    assert download.calls == [] and scorer.calls == []
    cache = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))
    assert set(cache) == {"1|the sound of rain|fake-clap", "2|the sound of rain|fake-clap"}


def test_choose_downloads_only_unscored(tmp_path):
    s = slot()
    sounds = [snd(id=1), snd(id=2)]
    run_choose(tmp_path, [s], {s.key: sounds[:1]}, {s.key: sounds[:1]}, points={1: 0.2})
    choices, scorer, download = run_choose(
        tmp_path, [s], {s.key: sounds}, {s.key: sounds}, points={2: 0.1}
    )
    assert download.calls == [2]
    assert choices[0].sound.id == 1  # キャッシュの 0.2 > 0.1


def test_choose_skips_failed_download_and_errors_if_all_fail(tmp_path):
    s = slot()
    sounds = [snd(id=1), snd(id=2)]
    choices, _, _ = run_choose(
        tmp_path, [s], {s.key: sounds}, {s.key: sounds}, points={2: 0.4}, download=FakeDownload(fail={1})
    )
    assert choices[0].sound.id == 2
    choices, _, _ = run_choose(
        tmp_path / "b", [s], {s.key: sounds}, {s.key: sounds}, download=FakeDownload(fail={1, 2})
    )
    assert choices[0].sound is None and choices[0].reason == "error"


def test_choose_score_cache_key_includes_clap_text(tmp_path):
    a = slot(sc=0)
    b = Slot(0, 1, "ambient", "rain", "[30 TO 900]", "the sound of storm", "storm")
    sounds = [snd(id=1)]
    _, scorer, _ = run_choose(
        tmp_path, [a, b], {a.key: sounds, b.key: sounds}, {a.key: sounds, b.key: sounds}, points={1: 0.3}
    )
    assert len(scorer.calls) == 2  # 文が違えば同じ音でも採点し直す


# ---------- selection / missing / exclude ----------


def choice(chapter=0, scene_i=0, kind="ambient", sound=None, reason=None):
    return Choice(chapter, scene_i, kind, "q", sound, 0.5 if sound else None, reason)


def test_selection_roundtrip_replaces_chapter_and_drops_others(tmp_path):
    path = tmp_path / "selection.json"
    write_selection(path, [choice(0, sound=snd(id=1)), choice(1, sound=snd(id=2))], {0, 1}, {0, 1})
    assert [c.sound.id for c in load_selection(path)] == [1, 2]
    # 章 1 だけ作り直し、章 0 は残す
    write_selection(path, [choice(1, sound=snd(id=9))], {1}, {0, 1})
    assert [(c.chapter, c.sound.id) for c in load_selection(path)] == [(0, 1), (1, 9)]
    # 小説から章 0 が消えた
    write_selection(path, [choice(1, sound=snd(id=9))], {1}, {1})
    assert [(c.chapter, c.sound.id) for c in load_selection(path)] == [(1, 9)]


def test_selection_processed_chapter_without_slots_clears_old_choices(tmp_path):
    path = tmp_path / "selection.json"
    write_selection(path, [choice(0, sound=snd(id=1)), choice(1, sound=snd(id=2))], {0, 1}, {0, 1})
    # 章 0 を処理したが枠が無かった（choices に章 0 が無い）→ 古い音は残さない
    write_selection(path, [], {0}, {0, 1})
    assert [(c.chapter, c.sound.id) for c in load_selection(path)] == [(1, 2)]


def test_load_selection_missing_file(tmp_path):
    assert load_selection(tmp_path / "none.json") == []


def test_write_missing_skips_intentional_silence(tmp_path):
    scenes = {0: [scene(0, ambient="rain"), scene(1, ambient=None, music="drone")]}
    choices = [
        choice(0, 0, "ambient", reason="no_result"),
        choice(0, 1, "bridge", reason="all_rejected"),
        choice(0, 0, "intro", sound=snd(id=1)),
    ]
    path = tmp_path / "missing.tsv"
    write_missing(path, choices, scenes)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "chapter\tscene\tstart_para\tkind\tdescription\tquery\treason"
    assert len(lines) == 3  # 選べた intro は出ない
    assert lines[1].split("\t") == ["1", "0", "0", "ambient", "scene 0", "q", "no_result"]
    assert lines[2].split("\t")[3:] == ["bridge", "scene 1", "q", "all_rejected"]
    # ambient_query が None の場面の ambient は（万一 choice があっても）出さない
    write_missing(path, [choice(0, 1, "ambient", reason="no_result")], scenes)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_read_exclude_ignores_comments_and_blank(tmp_path):
    path = tmp_path / "exclude.txt"
    path.write_text("# header\n\n123  # 違う音\n  456\n", encoding="utf-8")
    assert read_exclude(path) == {123, 456}
    assert read_exclude(tmp_path / "none.txt") == set()


def test_gate_schema_forbids_additional_properties():
    from irodori_speaker.bgm_select import GATE_SCHEMA

    assert GATE_SCHEMA["additionalProperties"] is False
    assert GATE_SCHEMA["properties"]["results"]["items"]["additionalProperties"] is False


def test_write_missing_lists_llm_error_scenes_in_order(tmp_path):
    bad = scene(1, ambient=None, music="")
    bad.status = "llm_error"
    scenes = {0: [scene(0), bad, scene(2, ambient=None, music="")], 1: [scene(0, chapter=1, ambient=None, music="")]}
    # 場面 2 は status ok の意図した無音なので出さない。場面 0 の choice は場面 1 の行より前に並ぶ
    choices = [choice(0, 2, "bridge", reason="no_result"), choice(0, 0, "ambient", reason="no_result")]
    path = tmp_path / "missing.tsv"
    write_missing(path, choices, scenes)
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    assert [(r[0], r[1], r[3], r[6]) for r in rows] == [
        ("1", "0", "ambient", "no_result"),
        ("1", "1", "ambient", "llm_error"),
        ("1", "2", "bridge", "no_result"),
    ]
    assert rows[1] == ["1", "1", "10", "ambient", "", "", "llm_error"]
