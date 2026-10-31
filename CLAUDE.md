# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

小説テキストを Irodori-TTS で読み上げる Python パッケージ `irodori_speaker`。感情は Ollama（既定）で句ごとに推定し、Irodori-TTS が解釈する絵文字として本文に挿入する。ユーザー向けの使い方は README.md（日本語）、設計は `docs/superpowers/specs/2026-10-24-irodori-speaker-design.md` にある。

## Commands

開発マシン（Mac）には torch / `irodori_tts` / GPU が無い。テストと `script` ステップはモデル無しで動くように作ってある。

```bash
# 依存関係とテスト（全体 / 1 ファイル / 1 件）
uv sync --extra dev
uv run pytest -q
uv run pytest tests/test_emotion.py -q
uv run pytest tests/test_pipeline.py::test_parse_chapter_range -q

# LLM 無しで分割と script.tsv の出力のみ（out/sample_novel/ に script.tsv と chapters.json）
uv run python -m irodori_speaker script tests/sample_novel.txt -c config.example.yaml --no-emotion

# WebUI の画面確認（生成には torch + irodori_tts + GPU が必要）
uv run --with gradio python -m irodori_speaker.webui
```

実際の合成は GPU マシン（WSL2）で行う。`--frozen` で Mac と同じ `uv.lock` を使い、以後は `--no-sync` で実行する。FFmpeg が必要（torchcodec）。

```bash
uv sync --frozen --extra tts --extra cu128 --extra webui
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml     # ステップ 1（Ollama が要る）
uv run --no-sync python -m irodori_speaker synth  novel.txt -c config.yaml     # ステップ 2（TTS だけ）
uv run --no-sync python -m irodori_speaker run    novel.txt -c config.yaml     # 一括
uv run --no-sync python -m irodori_speaker.webui -c config.yaml --host 0.0.0.0
uv run --no-sync python -m irodori_speaker bgm    novel.txt -c config.yaml     # ステップ 3（synth 済みに環境音と音楽を重ねる。Ollama・Freesound のキー・CLAP が要る）
uv run --no-sync python -m irodori_speaker check  novel.txt -c config.yaml     # 合成済みの文を調べて check.tsv に印（--asr で kotoba-whisper の文字誤り率も）
uv run --no-sync python -m irodori_speaker export novel.txt -c config.yaml     # iPhone で聴く形式に書き出す（既定 m4b、--format mp3、--bgm で BGM 付き）
# キャプション（上流の SamplingRequest.caption）で感情を指示できるかの実験。本体の合成はまだキャプションを使わない
uv run --no-sync python scripts/caption_test.py -c config.yaml --checkpoint <A> --checkpoint <B>
```

サブコマンドは `script` / `synth` / `run` / `bgm` / `check` / `export` / `config`。`config --update -c config.yaml [--example PATH] [--dry-run]`（novel を取らないので `main` は `load_config` の前に分岐する）は、`config.example.yaml` にあって `config.yaml` に無い最上位のキーと `_SECTIONS` の節の子キーだけを、例のコメントごとテキストの行として差し込む（`config_update.update_config_text`。値・コメント・並び・改行（CRLF）・権限・シンボリックリンクを保ち、`.bak` を残す。足した後に読み直して `asdict` が変わらなければ書き込む。例の値が既定値と型まで同じでないキー（今は `voice`）は足さずに「手で確かめて」と表示する）。`script` は `--chapters --no-emotion --reannotate --llm-model --think`、`synth` は `--chapters --force --voice --checkpoint --lora-adapter`、`run` はその両方を取る。`bgm` は `--chapters --replan --llm-model --think --ambient-db --music-db`（`run` には入れない）。`check` は `--chapters --asr --voice --checkpoint --lora-adapter`、`export` は `--format --bitrate --bgm`（常に全章。`--chapters` は無い）。設計は `docs/superpowers/specs/2026-10-31-listening-features-design.md`（ただし自動の題は `AUTO_TITLES` ではなく `_split_chapters_marked` の旗で判定する）。lint / formatter の設定は無い。`uv.lock` の `irodori-tts` を更新するときは `uv lock --upgrade-package irodori-tts`。

## Architecture

2 つのステップ（と、その後に独立して走る `bgm`）に分かれ、`cli.py` と `webui.py` は `pipeline.py` の同じ関数（`build_script` / `synthesize_book`）を呼ぶ別入口。ステップ 2 は `script.tsv` と `chapters.json` だけを読み、LLM に触れない。

1. `text_splitter.py`: 青空文庫のルビ・注記を読みに置換 → 章（正規表現）→ 段落（改行）→ 文（。！？、括弧の内側では切らない、1 文は `split.max_chars`=40 まで。長い文は後半で参照音声の声から離れるため短くしている）。`Segment(chapter_idx, para_idx, seg_idx, kind, text)` が全工程のキー。`kind` は `narration` / `dialogue` / `title` / `break`。`read_chapter_title`（既定 true）なら見出しの章は段落 0 が `title` の 1 文で本文は段落 1 から（自動の題「序」「全編」は `_split_chapters_marked` の旗で判定して読まない）。`is_separator`（文字・括弧・絵文字を含まない行。U+1F000 以上か U+FE0F を含む行は絵文字として区切りにしない）は `break` の 1 文になり、連続は 1 つ、本文の先頭・末尾の区切りは捨てる。
2. `emotion.py`: `split_clauses` が文を句（`、。！？!?…‥，,`）に分け、**段落単位**で LLM を呼んで句ごとに感情 10 種 × 強度 1〜3 を得る。出力は `validate()` で検証し、不正値・欠落は neutral にフォールバック。成功結果だけ `annotations.jsonl` にキャッシュする（キーはモデル・段落・前後の文脈・`think`）。`apply_shift` で地の文の強度を下げ、`build_spoken_text` が `emotion_emoji` の表から絵文字を句の先頭（括弧で始まる句は括弧の直後）に置く。同じ文の中で直前の句と同じ絵文字は置かない。`LLMAnnotator` は `llm.api` で 2 系統（`ollama` / `openai`）、`LLMServer` が `server_command` のプロセスを起動・停止する。
3. `pipeline.annotate_chapters`: 結果を `out/<book>/script.tsv` に書く。LLM の起動・停止と `unload()` はこの関数の中で完結し、ステップ 2 に LLM の状態を持ち込まない。`build_script` が `chapters.json` も書く。`title` / `break` の段落は LLM に渡さず、前後の文脈（`annotations.jsonl` のキーに入る）にも入れない（`title` は `neutral:1` で `text` = 見出し、`break` は `labels` / `text` とも空）。
- `emoji.py`: Irodori-TTS が解釈する絵文字 45 種（`IRODORI_EMOJI`）。`emoji_runs` で本文中の連なりを判定し、`safe_cut` で文字数での分割が連なりの途中を切らないようにする。`webui.carry_emoji` と `text_splitter` が使う。
4. `voices.py`: `voices/<id>.<ext>` か `voices/<id>/`（複数クリップは名前順）を解決し、`voice_key` はファイル名と中身の sha1（mtime は使わない）。予約語 `NO_REF_VOICE = "none"`（`is_no_ref`）は参照音声なしを表し、`resolve_voice` を通さない。
5. `synth.py`: `check_vram` → `load_runtime` → `encode_voice` → `Synthesizer.synthesize(text, force, seed=None)`。文ごとに `segments/<Synthesizer.segment_key(text, seed)>.wav` に保存する。`voice_paths` が空なら参照音声なしで、`latent_key` は `codec_repo` だけを含む `no_ref_latent_key`、`latent_path` は None（encode しない）、`_request` は `ref_latent=None, no_ref=True` を渡す。`_request` はモデルに渡す直前に読み辞書（`readings.apply_readings`、最長一致の 1 パス置換）→ `tts_text` を 1 回だけ当て、`lora_path` があるときだけ `lora_adapter` を渡す。`seed` が None のときは `_request` に渡さない（`params.update` で config の seed が None に化けるため）。
- `readings.py`: `load_readings(cfg, out_dir)` は config の `readings`（YAML のパス）→ `out/<book>/readings.yaml`（後勝ち）。`漢字表記: よみ` の平らな mapping、構文の誤りも日本語の ValueError。`script.tsv`・`source`・`warnings.log` は生の本文のまま。
- `lora.py`: `lora_adapter`（未統合の話者 LoRA のアダプタディレクトリ。`adapter_config.json` + `adapter_model.safetensors|bin`）を `Synthesizer.__init__` で runtime に触る前に検証する。`lora_key` は中身の sha1。上流は一度読んだアダプタをモデルから外さないので、`load_runtime` は `(RuntimeKey, lora_key)` が変われば（外す場合も）runtime ごと読み直す。WebUI は `Engine.get` で `refresh_lora`（stat が変わったときだけ sha1 を取り直す）。
6. `pipeline.synthesize_book` + `assemble.py`: 文間・段落間・章頭の無音を挟んで章 WAV（`chapters/NN_タイトル.wav`）、続けて全体 WAV を書く。サンプリングレートは固定せず、キャッシュ WAV から読む。`break` 行は合成せず（`text` に何があっても無視）`Clip(None, …, kind="break")`。間は `assemble.gap_seconds` が基本（文 / 段落）と特別（`scene`: 区切りの位置、`title`: 章題の後、`turn`: 同じ段落の台詞⇔地の文、`ellipsis`: 前の文が `…‥―—` で終わる）の大きい方。`.times.json` は `paragraphs`（区切りは幅 0 の `[para, pos, pos]`。bgm の段落集合の事前検証のため）と `segments`（`[para, seg, start, end]`、サンプル単位、音声のある文だけ）。
7. `check.py` + `pipeline.check_book`: 合成済みの文（`segments/`）を torch 無しで調べ、`check.tsv` に印（`long` / `short`: 読み辞書適用後の文字数あたり秒数の robust z、`clip`、`tail`: 末尾の孤立した短い音、`silent`: 本文があるのに発話が 0.1 秒以下、`missing`、`warned`: `warnings.log` にある、`cer`: `--asr` のときだけ）。キーは `check.segment_key(synth, row)` = `synth.segment_key(text, seed=row_seed(row))` で、`synthesize_book` と同じ計算（読み辞書・LoRA・行 seed 込み）。ASR（`kotoba-tech/kotoba-whisper-v2.0`）は `transcribe` の中でだけ transformers を import し、`batch_size` ごとに読み込み、`chunk_length_s=30` で 30 秒超の文も扱う。認識する WAV が無ければモデルを読み込まない。印があっても終了コード 0。
8. `export.py` + `pipeline.build_export` + `audio_io.write_m4b / encode_file`: 章 WAV（`--bgm` は `bgm/chapters/*.mp3`）を m4b（AAC、ffmetadata の章マーカー、`-f ipod`）か mp3（章ごと + 通し）に書き出す。章マーカーは実際にエンコードする入力の frames から計算する（bgm の MP3 は intro / bridge の分だけ長い）。`<出力>.sig` は入力の `.sig`・stat・題名・形式・ビットレート・チャンネルから作り、一致すれば再エンコードしない。入力の検証・エンコードの失敗時と、書き出し中に入力が書き換わったとき（`input_stamps` の取り直し）は、通し・`.sig`・章ディレクトリの MP3 をすべて消す。成功時は今回に含まれない章 MP3 を消す。

### script.tsv

列は `chapter, para, seg, kind, labels, text, source, seed`。`source` は分割直後の原文（一致判定用で編集しない）、`labels` は句ごとの `感情:強度` を `|` で並べた参考情報、`text` が絵文字入りの読み上げ本文で、合成とキャッシュキーに使われる。`seed` は空なら `sampling.seed`、整数ならその文だけその seed で合成する（WebUI の「取り直し」が増やす）。`load_script` は `seed` 列の無い古いファイルも読む。

**手直しが正**。`source` が小説本文と一致する行は、保存済みの `labels` / `text` をそのまま使い（絵文字を作り直さず、shift も再適用しない）、段落の全行が一致すれば LLM も呼ばない。`--reannotate` だけが手直しを捨てて作り直す（`emotion_emoji` を変えた後に使う）。そのとき LLM を呼ぶのは `annotations.jsonl` に無い段落だけ。`--chapters` で一部だけ処理しても、他の章の行は残す。小説から消えた章の行・文は落とす。題や区切りの挿入で段落番号がずれても手直しを引き継ぐため、`_match_saved_rows` が 2 パス（キーと `source` の一致 → 同じ章で `source` が同じ未使用の行を文書順に。`(章, source)` ごとの索引）で一度だけ対応づけ、`kept` の判定と行の生成の両方に使う。手直しの `seed` も引き継ぐ。

### キャッシュ

- 文のキャッシュ（`synth.cache_key`）: 読み辞書を当てた `text`（絵文字込み、`tts_text` 前）・参照音声の潜在キー・checkpoint・`sampling` の各値（`seed` は行の seed があればそれ）のハッシュ。LoRA アダプタの `lora_key` は**値があるときだけ**入れる（既存のキャッシュを無効にしない。新しい項目を足すときもこの方針）。`device` / `precision` は含めない。設定を変えると自動で作り直しになり、中断後の再実行は続きから再開する。
- 潜在キャッシュ（`out/.voices/<latent_key>.pt`）: 参照音声を DACVAE で一度だけエンコードして保存し、以後は `ref_latent` で渡す。キーは参照音声の中身・`codec_repo`・`ref_normalize_db`・`max_ref_seconds`（`null` のときは checkpoint）。
- 章 WAV は `.sig` に署名（本文・`(para, seg)` の並び・`kind` の並び・サンプリング・無音・checkpoint・潜在キー・読み辞書と LoRA の識別子（どちらかがあるときだけ）・`TEXT_RULE`・行の seed（どれかの行にあるときだけ））を持ち、全体 WAV は全章が揃って署名が一致するときだけ作る。章 WAV と全体 WAV の書き込みは一時ファイル経由で、失敗時は .sig が消えて stale 扱いになる。止めたとき・失敗時は古い全体 WAV を消す。

`webui.py`（Gradio、`-c` / `--host` / `--port`、既定 `127.0.0.1:7860`）は別の入口で、`Engine` が (checkpoint, device, precision, voice, lora_adapter) の組ごとに `Synthesizer` を作り直し、変わると runtime を読み直す。`Engine.get(…, out_dir=)` は呼ぶたびに本ごとの読み辞書を差し替える。LoRA の欄は各 `.click` の inputs の末尾・ハンドラの末尾の kwarg（既存テストが位置引数で呼ぶため）。生成ハンドラは generator で状態メッセージを yield する。

- 「台本を作る」「一括実行」の台本部分と「LLM で絵文字を付ける」は、LLM を動かす前に TTS の runtime を GPU から解放する（8GB で両方を載せない）。TTS は次の生成で読み直される。GPU を使う操作は `concurrency_id="gpu"`（上限 1）で直列にする。
- 「音声にする」は表の編集を `script.tsv` に保存してから `load_book` → `check_vram` → `synthesize_book` を回す。表が空なら `script.tsv` をそのまま使い、無ければ案内して止まる。表で編集できるのは `text` と `seed` 列だけで、他の列は読み込み時の値を使う（`rows_from_dataframe`。`break` 行の `text` は常に空）。「一括実行」の合成は、直前に作った `script.tsv` をそのまま使う。
- 表の行を選ぶ（`select_row`）とその文のキャッシュ WAV を再生する。選択は Engine を書き換えない（`Engine.peek` と独立した `Synthesizer` でキーを引く。合成中の共有インスタンスの `segments_dir` 等を変えないため）。小説の切り替えで選択は消える。「取り直し」（GPU）は seed を 1 増やしてその文だけ合成し、表の `seed` を更新する（`script.tsv` は「音声にする」が保存する）。「チェック」は `check_book`（ASR なし、GPU なし）。「書き出し」は `build_export`（GPU のキューに入らない。書き出し中に章が書き換わると `build_export` が出力を捨てて止まる）。
- 「BGM を付ける」（GPU）は `release_tts` → `stream_progress` で `build_bgm`（章の範囲と画面の LLM 設定、環境音・音楽の音量スライダーを渡す。`--replan`・停止・`exclude.txt` の編集は CLI と手書きのまま）。例外は `bgm_error_message`（`KeyMissing` は作り方、`DailyLimit` は翌日の案内、`ImportError` は `cli.BGM_IMPORT_MISSING`）で `gr.Error` にする。結果は `bgm_summary` と通し MP3・章 MP3。
- 試聴の「生成」「総当たり生成」は、本文を `split_preview_text`（小説生成と同じ `split.max_chars` の分割）で文に分け、文ごとに `generate_once` を呼んで `join_audio` で `silence` の無音を挟んで結合する（長い文は後半で声が崩れるため）。
- 「停止」は `should_stop` のフラグを立てるだけで、現在の文が終わり次第止まる。文ごとのキャッシュがあるので、再実行で続きから進む。
- 「声を追加」は `voices/<ID><元の拡張子>` にコピーする。既存の ID は上書きせず拒否する。
- 絵文字パレット（`irodori_tts.gradio_emoji_palette`）は import できるときだけ使い、できなければ感情プリセットの Dropdown と「絵文字を文頭に挿入」ボタンにする。「総当たり生成」は同じ seed で文頭の絵文字だけを変えて 10 感情を生成する。

### bgm ステップ

`bgm` は `synth` の後に走る独立した工程で（CLI の `bgm` と WebUI の「BGM を付ける」が同じ `build_bgm` を呼ぶ。`run` には入れない）、`script.tsv` / `chapters.json` / 章 WAV と `.times.json`・`.sig`（無ければ `synth --chapters N` を案内して止まる）だけを読み、朗読の WAV を書き換えない。入口は `cli._run_bgm` → `pipeline.build_bgm`（`BgmDeps` で LLM・Freesound・CLAP・サーバー・ダウンロードを差し替える。`bgm_*` / `emotion` は関数の中で遅延 import し、`pipeline.py` のトップレベルに置かない）。設計は `docs/superpowers/specs/2026-10-31-bgm-design.md`。

1. `build_bgm` の順序: `load_book` → 対象の章すべてで `bgm_mix.load_times`（無ければ、または `.times.json` の段落番号の並びが今の `script.tsv` と違えば `synth --chapters N` を案内して `ValueError`、LLM を起動する前。段落内の文言だけの変更は検出できない）→ `freesound.load_key`（LLM の前。`KeyMissing`）→ `with LLMServer`: `bgm_scenes`（`ScenePlanner.plan_chapter` → `merge_short` → `scenes.json`）、`bgm_select.search_slots` → `Gate` / `gate_slots` → `gate.tsv`、`unload` → **LLM を解放してから** `ClapScorer` を作り `choose` → `selection.json` / `missing.tsv` → 章ごとに `build_layout` → `render`（同じ音は章の中で 1 回だけデコード）→ `write_mp3` → `.sounds.json` → `.sig` → 通し。`CREDITS.txt` は `build_bgm` の終わりと失敗時（finally）に、`bgm/chapters/` にある `.sounds.json` 付きの全 MP3（章の題名を変えて旧名で残ったものも。titles は使わない）の音の和集合から作り直す（`_write_credits_from_mp3s`）。`missing.tsv` は `--chapters` の一部処理でも他の章の行を消さないよう、書き終えた `selection.json` / `scenes.json` の全章分から作る。
2. 8GB の GPU に LLM と CLAP を同時に載せない。`ClapScorer` は LLM の `unload` の後に作り、選び終えたら捨てる。torch / transformers は `ClapScorer` の中でだけ import する。
3. キャッシュ（`bgm/cache/`）: `scenes.jsonl`（場面）・`search.json`（検索語と条件ごと。上限で止まった後の再実行で API を呼ばない）・`gate.jsonl`（ふるい分け）・`scores.json`（CLAP の点）。`--replan` は `scenes.jsonl` だけ消す。試聴 MP3 は本をまたぐ共有キャッシュ `<output_dir>/.freesound/previews/<ID>.mp3`（`freesound.cached_preview`、あればダウンロードしない。採点用も重ね合わせ用も同じ。消さないが、CLAP の採点や `decode` で読めなかったファイルは消して次の実行で取り直す）。章の MP3 は `<mp3>.sig`（朗読の署名・場面・選んだ音の ID・`bgm` 設定の `mix_signature`）が一致し、`<mp3>.sounds.json`（その MP3 で実際に使った音）があれば重ね直さない。重ね直しに失敗した章は MP3・`.sig`・`.sounds.json` を消す（CREDITS.txt と食い違う古い MP3 を残さない）。通しは `--chapters` なしで全章の MP3 が揃ったときだけ作り、それ以外と失敗時は古い通しを消す。`--chapters` で一部だけ処理しても `selection.json` / `scenes.json` の他の章は残す。
4. 1 日の上限（`DailyLimit`）は捕まえずに上げ、MP3 は書かない（`cli` は終了コード 1）。`KeyMissing` / `ImportError` は終了コード 2、`ValueError` / `RuntimeError` は 1。
5. キーは `~/.config/irodori-speaker/secrets.env`（`bgm.secrets_file`）の `FREESOUND_API_KEY=` 行だけに置く。環境変数に置かない。`Authorization` ヘッダーで API ホストにだけ送り、URL・例外・ログ・キャッシュに含めない（試聴 MP3 の取得には付けない）。**実装・テストはこの実ファイルを読まない**（テストは `tmp_path` に偽のファイルを作る）。
6. 音量（`bgm_mix.render`）: 声と素材を `loudness_db`（BS.1770 の K 特性を 400 ms ブロックの rfft パワーにかける簡略版。絶対ゲート -70・相対ゲート -10、オフセット -0.691）で測り、素材を「声の音量 + `ambient_db` / `music_db`」に合わせる。素材はステレオ（L+R の和）、モノラルの声は両チャンネルに入るので +3.01 dB して揃える。規則を変えたら `MIX_RULE` を上げる（今は 2）。`ambient_db` / `music_db` は CLI の `--ambient-db` / `--music-db` と WebUI のスライダー（範囲は `bgm_slider_range` で config の値を含む）で、config.yaml を書き換えずにその実行だけ上書きできる（ユーザーの config.yaml は直しにくいので、調整値は既定値の変更より上書きの口を用意する）。
7. `exclude.txt`（手で書く Freesound の ID、`bgm` は読むだけ）以外の `scenes.json` / `selection.json` は派生ファイルで、手で直さない。出力は私的な利用に限る（小説の著作権と Freesound のライセンス）。

## 注意点

- 文末の句点・読点（閉じ括弧の直前を含む）は、モデルに渡す前に `synth.tts_text` で外す（句点・読点で終わると文末に余計な音が入るため）。台本の `text` は変えない。加工規則を変えるときは `TEXT_RULE` を更新する（文のキャッシュと章署名に入る）。
- `torch` と `irodori_tts` は `synth.py` / `webui.py` 内（と `cli._make_synthesizer`）で遅延 import している。トップレベルに移すと Mac でのテストと `script` が壊れる。`emotion.py` も `pipeline.annotate_chapters` の中で遅延 import している（`--no-emotion` を動かすため）。TTS の import に失敗したら `cli.py` が案内を出して終了コード 2 で止まる。
- **8GB の GPU では LLM と TTS を同時に載せない**。`run` は台本 → LLM 解放 → 音声化の順。`load_runtime` はモデルを読み込む前に `check_vram`（`min_free_vram_gb`）で空きを確認し、足りなければ LLM サーバーの停止を案内する。モデルを切り替えるときは旧 runtime を先に解放してから確認する（上流は新旧を同時に持つため）。
- LLM の `ollama` は OpenAI 互換ではなく**ネイティブ `/api/chat`** を使う。`think: false` と `format`（JSON Schema）、`keep_alive: 0` が確実に効くため（OpenAI 互換経由では思考モードの無効化が届かない報告がある）。`openai` は `response_format`（400 なら `json_object`、さらに 400 なら余分なキーを外す）と `chat_template_kwargs` / `reasoning_effort` を送り、`<think>` は除去する。
- **絵文字の配置（句の先頭、連発しない、文の先頭は必ず置く）は本ツールの仮定**。上流の `EMOJI_ANNOTATIONS.md` は配置ルールを定めていない。変更時は WebUI の「総当たり生成」で効きを確認する。強度 1 は絵文字なし、強度 3 は同じ絵文字の繰り返し。😭 のような非言語音の絵文字は発話の前に泣き声が入ることがあるので、既定では sad の強度 3 だけに使う。
- 長文は不安定になり、`sampling.max_seconds`（30 秒）で切り詰められるので、文単位で合成して結合する設計を崩さない。切り詰めは `res.messages` の `info: predicted duration …` を `synth.collect_warnings` が読んで `warnings.log` に記録する。
- 参照音声（`ref_latent`）に対応しないチェックポイント（VoiceDesign 等）は対象外で、`load_runtime` が ValueError にする。
- `voice: none` は話者 LoRA 統合 checkpoint 向けの参照音声なし合成（ユーザー決定 2026-10-25）。`null` は none の意味にせず、従来どおりエラーで止める。`voices/none.wav` があっても none は参照音声なしとして扱い、WebUI の「声を追加」は ID `none` を拒否する。声の候補は `webui.voice_choices`（先頭が none）。ベースモデルでは声が定まらない。
- 出力には上流の仕様で SilentCipher の透かしが入る。他人の声を同意なく複製しないこと（モデルカードの倫理規定）。
- `voices/` と `config.yaml`、`out/` は git に入れない（`.gitignore`）。
