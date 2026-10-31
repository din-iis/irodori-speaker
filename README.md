# irodori-speaker

小説のテキストファイルを、[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（日本語特化のオープンソース TTS）で読み上げます。
声は参照音声（ゼロショットの声クローン）から作り、感情はローカル LLM（Ollama）が句ごとに推定して、Irodori-TTS が解釈する絵文字として本文に挿入します。

- 合成は GPU マシン（Windows + WSL2、RTX 3060 Ti 8GB を想定）で行います。Mac は開発・テストと、LLM だけを使う `script` ステップ用です。
- 設計: `docs/superpowers/specs/2026-10-24-irodori-speaker-design.md`

## 仕組み

処理は独立した 2 つのステップに分かれています。

1. **台本（`script`）**: 小説を章・段落・文に分け、LLM が句ごとの感情を推定して、絵文字入りの読み上げ本文を `script.tsv` に書き出します。LLM だけを使い、torch は要りません。終わると LLM を VRAM から解放します。
2. **音声化（`synth`）**: `script.tsv` の `text` 列を文ごとに Irodori-TTS で合成し、章ごとの WAV と全体の WAV に結合します。TTS だけを使い、LLM には接続しません。

`run` は 1 → 2 を続けて実行します。8GB の GPU には LLM と TTS を同時に載せられないので、台本を作り終えて LLM を解放してから TTS を読み込みます。2 つを分けてあるので、台本を手で直してから音声化できます。

### 句読点ごとの絵文字

Irodori-TTS は、本文中の絵文字（😊 😠 👂 など）で演技を制御します。文頭に 1 個だけ置くのではなく、**句読点（「、。！？…」）で区切った句ごとに**感情を推定して絵文字を置きます。

```
入力:   「やった！ついに見つけたぞ！！」
台本:   「😲やった！😆ついに見つけたぞ！！
```

（実際の絵文字は LLM の推定と `emotion_emoji` の表で決まります。上は例です。）

- 感情は 10 種（neutral / happy / sad / angry / surprised / fearful / tender / whisper / excited / serious）× 強度 1〜3 です。`emotion_emoji` の表で絵文字に変換します。強度 1 は絵文字なしです。
- 地の文は強度を `narration_intensity_shift`（既定 1）だけ下げます。地の文はほぼ素の朗読になり、台詞に感情が乗ります。
- 絵文字は句の先頭（句が「『（で始まるなら開き括弧の直後）に置きます。同じ文の中で直前の句と絵文字が同じなら置きません。文は別々に合成するので、文の先頭の句には必ず置きます。
- この配置は本ツールの仮定です（上流の絵文字の説明は配置ルールを定めていません）。効きは WebUI の「感情プリセット総当たり」で確かめてください。

## セットアップ

### GPU マシン（合成）

前提: Windows + WSL2（Ubuntu など）、NVIDIA GPU（CUDA 12.8 対応のドライバー）、[uv](https://docs.astral.sh/uv/)。

```bash
# 1. FFmpeg（torchcodec が必要とする）
sudo apt install ffmpeg

# 2. 依存関係（Mac と同じ uv.lock のコミットを使うため --frozen）
uv sync --frozen --extra tts --extra cu128 --extra webui

# 3. WSL2 側に Ollama と LLM（Q8_0 で約 5.2GB）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:4b-q8_0

# 4. 設定
cp config.example.yaml config.yaml    # voice などを編集
```

以後は `uv run --no-sync python -m irodori_speaker ...` で実行します（`--no-sync` を付けると、`uv run` が環境を同期し直さず、上で入れた GPU 用の torch をそのまま使います）。

初回の合成時に、Irodori-TTS のチェックポイントとコーデック（`Aratako/Semantic-DACVAE-Japanese-32dim`）を Hugging Face から自動でダウンロードします。

### 参照音声（声）の置き方

`voices/` に参照音声を置き、`config.yaml` の `voice` で選びます。次のどちらでも構いません。

```
voices/narrator.wav          # 1 ファイル → voice: narrator
voices/narrator/             # フォルダ内の複数クリップ → voice: narrator
  01.wav                     #   名前順に連結して 1 つの声にする
  02.wav
```

- 対応する拡張子: `.wav .flac .mp3 .m4a .ogg .opus .aac .webm`
- 1 つの声で地の文も台詞も全部読みます（登場人物ごとの声の割り当ては未対応です）。
- 参照音声は初回の合成で一度だけ潜在表現（`out/.voices/<sha1>.pt`）に変換してキャッシュします。ファイル名と中身が同じなら、`voices/` を rsync し直してもキャッシュは有効です。
- `voices/` の中身は git に入れません（`.gitignore` 済み）。GPU マシンに直接置くか、rsync で送ってください。

### 参照音声を使わない（`voice: none`）

`voice` に予約語 `none` を書く（CLI は `--voice none`、WebUI は声の欄の先頭の「none」）と、参照音声を渡さずに合成します（上流の `no_ref=True`）。

- **話者 LoRA を統合した checkpoint と組み合わせる設定です。** v4.1-Small は参照音声を渡すと、絵文字でもキャプションでも感情がほとんど出ませんでした。話者 LoRA を統合した checkpoint を参照音声なしで使うと、声が文ごとにぶれず、絵文字の感情も効きました。
- **ベースモデルでは使わないでください。** 参照音声がないと声が定まらず、文ごとに別人になります。
- 参照音声の変換（潜在表現のキャッシュ）は行いません。文のキャッシュと章 WAV の署名は参照音声ありと別になるので、`none` に切り替えると作り直しになります。
- `none` は予約語です。`voices/none.wav` があっても参照音声としては使われず、WebUI の「声を追加」も ID `none` を拒否します。`voice` を未指定（`null`）にしたときは、従来どおり案内を出して止まります。

### Mac（開発・テスト）

Mac には torch も GPU も入れません。`script` ステップ（Ollama があれば LLM 付きで、無ければ `--no-emotion`）とテストが動きます。

```bash
uv sync --extra dev
uv run pytest -q
```

WebUI の画面だけ確認するなら（生成はできません）:

```bash
uv run --with gradio python -m irodori_speaker.webui
```

## 使い方（CLI）

どのサブコマンドも `novel.txt` を受け取り、作業ディレクトリは `out/<novel の拡張子を除いた名前>/` です。順に実行してください。

```bash
# 1. 台本: 分割 + 句ごとの感情 → out/<book>/script.tsv, chapters.json
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml

# （script.tsv の text 列を手で直す）

# 2. 音声化: script.tsv → 章ごとの WAV と全体の WAV
uv run --no-sync python -m irodori_speaker synth novel.txt -c config.yaml

# 一括: 1 → 2 を続けて実行（間で LLM を解放してから TTS を読み込む）
uv run --no-sync python -m irodori_speaker run novel.txt -c config.yaml
```

`script` は先頭に「N 章 / N 段落 / N 文」を、`synth` は使う声（クリップ数と合計秒数）を表示し、終了時に `warnings.log` の件数を表示します。

### オプション

共通（`script` / `synth` / `run`）:

| オプション | 内容 |
|---|---|
| `-c`, `--config` | 設定 YAML。省略すると組み込みの既定値 |
| `--chapters 1-3,5` | 対象の章（1 始まり） |

`script` と `run`:

| オプション | 内容 |
|---|---|
| `--no-emotion` | LLM を使わず全て neutral にする（`text` = 原文）。Mac で LLM なしに動かすときに使う |
| `--reannotate` | script.tsv の手直しを捨てて、感情と絵文字を作り直す |
| `--llm-model NAME` | LLM のモデル名（config の `llm.model` を上書き） |
| `--think` | LLM の思考モードを有効にする（精度は上がるかもしれないが数倍遅い） |

`synth` と `run`:

| オプション | 内容 |
|---|---|
| `--force` | 合成済みの文も作り直す |
| `--voice ID` | 声の ID（config の `voice` を上書き）。`voices/` 内の ID のほか、ファイルやフォルダのパスも指定できる。`none` なら参照音声を使わない |
| `--checkpoint ID` | TTS のチェックポイント（config の `checkpoint` を上書き） |

```bash
# 例: 1〜3 章だけ台本を作る / 別の LLM を使う / 声を変えて作り直す
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml --chapters 1-3
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml --llm-model qwen3.5:4b-q4_K_M
uv run --no-sync python -m irodori_speaker synth novel.txt -c config.yaml --voice narrator2 --force
```

`--help` で各サブコマンドのオプションを確認できます。

### 出力

```
out/
  .voices/<sha1>.pt            参照音声の潜在表現のキャッシュ
  <book>/
    chapters.json              章のタイトル（script が書き、synth が読む）
    script.tsv                 台本。手で直すと次回の音声化に反映される
    annotations.jsonl          LLM の推定結果のキャッシュ（成功した段落のみ）
    segments/<sha1>.wav        文ごとの合成結果のキャッシュ（途中で止めても続きから再開できる）
    warnings.log               合成時の警告（文が長すぎて切り詰められた等）
    chapters/NN_タイトル.wav   章ごとの WAV
    <book>_full.wav            全体の WAV
```

`--chapters` で一部の章だけ合成したときは、全章の WAV が揃っていて台本と一致している場合にだけ `<book>_full.wav` を作ります。揃っていなければ古い通し WAV は削除されます。

### script.tsv を手直しする

`script.tsv` はタブ区切りで、列は次のとおりです。

| 列 | 内容 |
|---|---|
| `chapter` / `para` / `seg` | 章・段落・文の番号（0 始まり）。行のキー |
| `kind` | `narration`（地の文）か `dialogue`（台詞） |
| `labels` | 句ごとの `感情:強度` を `\|` で並べた参考情報（例: `happy:2\|neutral:1`）。音声には使われない |
| `text` | **絵文字入りの読み上げ本文。手直しはここで行う** |
| `source` | 分割直後の原文。小説本文との一致判定に使うので**編集しない** |

`text` 列では、絵文字の移動・削除・追加と、読み仮名の修正ができます。直したら `synth` を再実行すると、**直した文だけ**が再合成されます（文ごとのキャッシュは `text` を含むキーで引くため）。

**手直しが正です。** `script` を再実行しても、`source` が小説本文と一致する行は、保存済みの `labels` と `text` をそのまま使います（絵文字を作り直さず、段落の全行が一致するなら LLM も呼びません）。小説本文を直した文だけが再推定されます。`--chapters` で一部の章だけ処理しても、他の章の行は残ります。

`emotion_emoji` の表を変えて、その結果を台本に反映したいときは `--reannotate` を付けます。手直しを捨てて、感情と絵文字を作り直します。LLM の推定は `annotations.jsonl` にキャッシュされているので、LLM を呼ぶのはキャッシュに無い段落だけです。

```bash
uv run --no-sync python -m irodori_speaker script novel.txt -c config.yaml --reannotate
```

## BGM を付ける

`synth` で作った朗読に、Freesound から自動で選んだ環境音と音楽を重ねて、章ごとと通しの MP3 を作ります（`bgm` サブコマンド）。朗読の WAV は読むだけで、書き換えません。`run` には含まれず、WebUI にもありません。

流れは、LLM で章を場面に区切る → Freesound を検索する → LLM で候補をふるい分ける → LLM を解放する → CLAP で順位を付けて選ぶ → 重ね合わせ、です。合う音が見つからない場面は、ずれた音を流さず無音にして `missing.tsv` に残します。

### 前提

- `synth` を済ませ、章の WAV と `.times.json` があること。`.times.json` が無い章があれば、`synth --chapters N` を案内して止まります（文のキャッシュがあるので音声合成はやり直しません）。
- GPU マシンで `uv sync --frozen --extra tts --extra cu128 --extra bgm` を実行する。Ollama が動いていること（`script` と同じ）。FFmpeg が必要。
- Freesound の API キーを、**自分の端末で**ファイルに書く。キーを環境変数やシェルの設定ファイルに置かないのは、AI のエージェントが読めてしまうため。`bgm` はこのファイルだけを読み、キーを画面・ログ・キャッシュに書かない。ファイルが無いかキーの行が無ければ、手順を表示して終了コード 2 で止まる。

```bash
mkdir -p ~/.config/irodori-speaker
read -rsp 'Freesound API key: ' k; echo
printf 'FREESOUND_API_KEY=%s\n' "$k" > ~/.config/irodori-speaker/secrets.env; unset k
chmod 600 ~/.config/irodori-speaker/secrets.env
```

ファイルの場所は設定 `bgm.secrets_file` で変えられます。

### コマンド

```bash
uv run --no-sync python -m irodori_speaker bgm novel.txt -c config.yaml
uv run --no-sync python -m irodori_speaker bgm novel.txt -c config.yaml --chapters 1-3
```

| オプション | 内容 |
|---|---|
| `--chapters` | 対象の章（1 始まり）。一部の章だけのときは、通しの MP3 を作らない |
| `--replan` | 場面の区切りをやり直す。場面の説明が変わるので、ふるい分けと採点もやり直しになる |
| `--llm-model` / `--think` | `script` と同じ（config の `llm` を上書き） |

各段の結果は本ごとのキャッシュに残るので、途中で止めても再実行で続きから進みます。重ね合わせは、朗読・選んだ音・`bgm` 設定が同じ章を飛ばします。音量や長さは `config.yaml` の `bgm` 節で調整します。

### 出力

```
out/<book>/bgm/
  scenes.json                  章ごとの場面（段落の範囲、説明、検索語、雰囲気）
  selection.json               場面ごとに選んだ音の ID・名前・作者・URL・ライセンス・点
  gate.tsv                     ふるい分けの採否と理由（外し方の確認用）
  missing.tsv                  見つからなかった音（理由: no_result / all_rejected / llm_error / error）
  CREDITS.txt                  使った音の題名・作者・URL・ライセンス
  exclude.txt                  手で書く除外リスト（下記）
  chapters/NN_タイトル.mp3     章ごとの BGM 付き MP3（.sig は重ね直しの判定用）
  <book>_bgm_full.mp3          通しの MP3（全章が揃うときだけ）
  cache/                       検索・ふるい分け・採点・場面のキャッシュ
```

`bgm/tmp/` にダウンロードした音を一時的に置き、重ね終えたら消します。選んだ音の ID とメタデータだけを残し、重ね直すときは選んだ音だけをダウンロードし直します。

### 違う音を替える（exclude.txt）

聴いて「この音は違う」と思ったら、`CREDITS.txt` にある Freesound の ID を `out/<book>/bgm/exclude.txt` に 1 行に 1 つ書きます（`#` 以降はコメント）。次の `bgm` で、その音を選んでいた場面はふるい分けで残った次点を選び直し、次点が無ければ無音になります。除外した ID は同じ本の他の場面でも選びません。`bgm` はこのファイルを読むだけで書き換えません。

### 1 日の上限で止まったとき

Freesound の API は 1 日 2000 回までです。上限に達すると「1 日の上限に達しました」と表示して終了コード 1 で止まり、MP3 は書きません（無音だらけの MP3 を作らないため）。それまでの検索・ふるい分け・採点はキャッシュに保存してあるので、**翌日に同じコマンドをもう一度実行**してください。

### 私的な利用に限る

出力は私的な利用に限り、配布しないでください。小説の著作権があり、Freesound の音には CC0・CC BY・CC BY-NC などのライセンスがあるためです（クレジットは `CREDITS.txt`）。

## WebUI

試聴（短い文で声・絵文字・パラメータを確認）と、小説全体の生成の両方に使えます。GPU マシンで起動し、Mac のブラウザから開きます。

```bash
uv run --no-sync python -m irodori_speaker.webui -c config.yaml --host 0.0.0.0
```

| オプション | 内容 |
|---|---|
| `-c`, `--config` | 設定 YAML。画面の初期値（チェックポイント・声・LLM・サンプリングの値）になる。省略可 |
| `--host` | 既定は `127.0.0.1`。Mac のブラウザから開くには `0.0.0.0` |
| `--port` | 既定は `7860` |

Mac のブラウザで `http://<GPU マシンの IP>:7860` を開きます（`--host` を省略した場合は GPU マシン自身の `http://127.0.0.1:7860`）。

### TTS 設定（画面上部）

- チェックポイント（候補から選ぶか、HF の ID・ローカルのパスを自由入力）、device、precision、声を選びます。
- 声は「更新」ボタンで `voices/` の一覧を読み直せます。
- 「声を追加」: 「参照音声をアップロード」に音声を入れ、「声の ID」を書いて押すと、`voices/<ID>.<元ファイルの拡張子>` に保存して一覧に加えます（対応外の拡張子は `.wav` になります）。**既にある ID は上書きせず、エラーにします**。ID には `/` `\` を含められず、`.` で始められません。
- 「モデル読み込み」: TTS モデルを読み込み、参照音声を潜在表現に変換します。押さなくても、最初の生成時に自動で読み込みます。チェックポイント・device・precision・声を変えると、次の生成の前に読み直します。読み込み前に空き VRAM を確認し、足りなければ状態欄に案内が出ます。

### LLM 設定（折りたたみ）

- api（ollama / openai）、base_url、モデル、思考モード、並列数を指定します。ここでの値は config の `llm` を上書きして、台本作成と「LLM で絵文字を付ける」に使います。空欄の項目は config の値になります。
- 「一覧を取得」ボタンで、サーバーにあるモデルの一覧をモデル欄に入れます（Ollama は `ollama list` と同じ一覧）。一覧に無いモデル名も自由入力できます。

### 8GB での LLM と TTS の切り替え

「LLM で絵文字を付ける」「台本を作る（script.tsv）」「一括実行」の台本の部分では、LLM を動かす前に TTS のモデルを GPU から解放します。LLM と TTS を同時に載せないためです。解放した TTS は、次の生成のときに読み直します（そのぶん最初の生成に時間がかかります）。LLM は注釈のあとに解放します。

### タブ「試聴」

- 本文を入れて「生成」を押すと、音声と使用した seed・所要時間・警告が出ます。num_steps・cfg_scale_text・cfg_scale_speaker・duration_scale・seed（`-1` でランダム）を指定できます。
- 長い文は小説生成と同じく `split.max_chars` で分けて合成し（分けた文には直前の絵文字を引き継ぎ）、文の間・段落の間に `silence` の無音を挟んで 1 本にします。
- 絵文字の入れ方は 2 通りです。`irodori_tts` がある環境（GPU マシン）では Irodori-TTS の絵文字パレットが出ます。無い環境では、感情と強度を選んで「絵文字を文頭に挿入」を押します（「『（で始まる本文では括弧の直後に入ります）。
- 「LLM で絵文字を付ける」: 本文を LLM に通し、句ごとの絵文字入りの本文に置き換えます。
- 「総当たり生成」: 同じ seed・同じ本文で、文頭に 10 感情それぞれの絵文字（強度はスライダーで指定、既定 3）を付けて 10 本生成し、並べて聞き比べます。`emotion_emoji` の調整に使います。差が聞こえなければ、その絵文字は効いていません。

### タブ「小説生成」

小説ファイル（アップロード、またはパスを入力）、章の範囲、オプション（感情を付けない / 感情を作り直す（reannotate）/ 作り直す（force））を指定して、次のボタンを**それぞれ独立に**押せます。小説を選ぶと、既に `script.tsv` があれば表に読み込みます。

- **台本を作る（script.tsv）**: `script` と同じ処理です。結果は表に出ます。
- **音声にする**: `synth` と同じ処理です。表を編集していれば、**その編集を `script.tsv` に保存してから**合成します。表が空のときは、`script.tsv` をそのまま使います。`script.tsv` が無ければ「先に台本を作ってください」と案内して止まります。完了後に章ごとの WAV と全体の WAV を表示します。
- **一括実行**: 台本を作り、続けて音声にします。音声にする段階では、直前に作った `script.tsv` をそのまま使います（表の編集は使いません）。
- **停止**: 現在の文の合成が終わり次第、止まります。文ごとにキャッシュしているので、同じ設定で「音声にする」を押し直すと続きから進みます。一括実行で台本の作成中に押したときは、台本を作り終えたところで止まります。
- **script.tsv を保存**: 表の編集を `script.tsv` に書き戻します。編集できるのは `text` 列だけで、他の列は読み込み時の値のままです。

表の `text` 列の手直しは、CLI の `script.tsv` の手直しと同じ扱いです（前の節を参照）。


## 設定

`config.example.yaml` をコピーして `config.yaml` を作ります（`config.yaml` は git に入れません）。主なキーは次のとおりです。

| キー | 既定値 | 内容 |
|---|---|---|
| `checkpoint` | `Aratako/Irodori-TTS-v4.1-Small` | Hugging Face の ID / ID + サブフォルダ / ローカルの `.safetensors` |
| `codec_repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | コーデック |
| `device` / `precision` | `cuda` / `bf16` | `cuda`・`mps`・`cpu` と `fp32`・`bf16`。CUDA は bf16 が速く VRAM も半分 |
| `min_free_vram_gb` | `3.0` | TTS を読み込む前の空き VRAM の下限。足りなければ LLM の停止を案内して止まる（`0` で無効） |
| `empty_cache_interval` | `10` | N 文ごとに `torch.cuda.empty_cache()`（`0` で無効） |
| `voices_dir` / `voice` | `voices` / 既定なし（例: `narrator`） | 参照音声の置き場所と、使う声の ID。`none` は参照音声を使わない予約語（話者 LoRA 統合 checkpoint 向け）。`voice` が未設定だと `synth` は案内を出して止まる（`--voice` でも指定できる） |
| `ref_normalize_db` | `-16.0` | 参照音声のラウドネス正規化（`null` で無効） |
| `max_ref_seconds` | `null` | 参照音声の上限秒数。`null` はチェックポイントの既定（v4 は 120 秒） |
| `sampling.*` | | `num_steps`・`cfg_scale_text`（3.0）・`cfg_scale_speaker`（5.0）・`duration_scale`（0.8）・`t_schedule_mode`・`sway_coeff`・`max_seconds`（30.0）・`seed`（42） |
| `emotion_mode` | `emoji` | `emoji` か `none`（`none` は LLM を使わない） |
| `narration_intensity_shift` | `1` | 地の文の強度を下げる量 |
| `emotion_emoji` | | 感情 × 強度（1/2/3）→ 挿入する絵文字。空文字は挿入なし。要素は必ず 3 つ |
| `llm.*` | | 下記 |
| `split.max_chars` | `40` | 1 回に合成する最大文字数。これより長い文は読点の位置で分けて別々に合成する。長い文は後半で参照音声の声から離れやすいため短めにしている |
| `split.chapter_regex` | 「第N章/話/節」「プロローグ」「エピローグ」 | 章の見出しの正規表現。見出しが無ければ全体を 1 章にする |
| `silence.*` | 文 0.25 / 段落 0.6 / 章 1.0 | 結合時に挟む無音（秒） |
| `output_dir` | `out` | 出力先 |

- 設定キーの綴りを間違えると、起動時にエラーで止まります（不明なキーは無視されません）。
- `sampling.seed` は固定を勧めます（`null` だと文ごとに声がぶれます）。
- `sampling` の値・`checkpoint`・参照音声・文の本文のどれかを変えると、文ごとのキャッシュのキーが変わり、自動で作り直しになります。`device` と `precision` はキーに含まれません。

### 8GB で OOM になるとき

`precision: bf16` で OOM になる場合は、int8 版のチェックポイントに切り替えます。

```yaml
checkpoint: Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only
```

また、Ollama を止め忘れて空き VRAM が足りないときは、`synth` の読み込み前に案内が出て止まります。`ollama stop <モデル名>` で解放してから再実行してください。

### LLM を選ぶ・切り替える

既定は Ollama の `qwen3.5:4b-q8_0`（約 5.2GB、思考モード無効）です。日本語 LLM の評価（Nejumi Leaderboard 4）で 10B 未満のモデルの中では上位で、Q8_0 なら 8GB に載ります。軽くしたいときは `qwen3.5:4b-q4_K_M`（約 3.3GB）にします。`script --llm-model NAME` か WebUI の LLM 設定でも切り替えられます。

Ollama の代わりに llama-server / vLLM / LM Studio を使うときは、`llm.api: openai` にして `base_url` を OpenAI 互換の `/v1` に向けます。

```yaml
llm:
  api: openai
  base_url: http://localhost:8080/v1
  model: Qwen3.5-4B-Q8_0
  # server_command を書くと、台本作成の前にツールが起動し、終わったら停止する
  server_command: ["llama-server", "-m", "models/Qwen3.5-4B-Q8_0.gguf", "-ngl", "99", "-c", "16384", "-np", "4", "--port", "8080"]
  concurrency: 4          # llama-server の -np に合わせる
```

- `server_command` を省略すると、起動済みのサーバーを使います（停止もしません）。指定してサーバーがすでに応答していれば、起動しません。
- 注釈が終わるとモデルを解放します（`llm.unload: true`）。Ollama は `keep_alive: 0` を送り、`openai` は `server_command` で起動したサーバーを停止します。
- `llm.api_key` を指定すると `Authorization: Bearer` ヘッダーを付けます。

## 声の作り方

- 参照音声は、**複数の短いクリップで合計 30 秒程度**が推奨です（上限は 120 秒）。長すぎる 1 ファイルより、雑音の無い短いクリップを数本そろえるほうが安定します。
- クリップは `voices/<id>/` に入れると名前順に連結されます。順序を決めたいときは `01.wav`・`02.wav` のように番号を付けてください。
- BGM・残響・複数人の声が混ざっていない、一定の調子で話した音声を選びます。
- 日本語のみに対応しています。
- 声を変えたら（ファイルの中身を差し替えたら）、参照音声の潜在表現と文ごとのキャッシュが自動で作り直されます。

### キャプションで感情を試す（実験）

v4.1-Small は、参照音声と一緒にキャプション（「激しく怒って、強い口調で相手を問い詰めている」のような文）を受け取れます。本ツールの合成はまだキャプションを使いませんが、`scripts/caption_test.py` で効き方を試せます。GPU マシンで、Ollama を止めてから実行します。

```bash
uv run --no-sync python scripts/caption_test.py -c config.yaml \
  --checkpoint Aratako/Irodori-TTS-v4.1-Small --checkpoint /path/to/merged.safetensors
```

- 声・sampling・seed は `config.yaml` の値を使い、キャプション（なし / 怒り / 悲しみ / 喜び / ささやき）と `cfg_scale_caption`（3 / 5）だけを変えて、台詞と地の文の 2 文を生成します。比べるため、キャプションの代わりに `emotion_emoji` の強度 3 の絵文字を文頭に付けたものも作ります。
- `--checkpoint` を複数渡すと順に比べます。省略すると `config.yaml` の `checkpoint` を使います。
- `--no-ref` を付けると参照音声を渡さずに生成します。参照音声があると感情が出にくいのかを切り分けるためのもので、ベースモデルでは声が参照音声と別人になります。
- 出力は `out/caption_test/<番号>_<checkpoint 名>/`（`--no-ref` のときは末尾に `_noref`）に、`1_none.wav`（1 文目・キャプションなし）、`1_angry_cap3.wav`、`1_angry_emoji.wav` のような名前でできます。文とキャプションはスクリプト冒頭の `TEXTS` と `CAPTIONS` で変えられます。

## 注意

- **透かし**: 出力される音声には、上流の仕様で SilentCipher の透かしが入ります。
- **声の同意**: 他人の声を、本人の同意なく複製しないでください（モデルカードの倫理規定）。参照音声には、自分の声か、利用許諾を得た声だけを使ってください。
- **漢字の読み間違い**: 青空文庫形式のルビ（`|漢字《かんじ》`）で小説本文に読みを書くか、`script.tsv` の `text` 列を直してください。ルビは読み仮名に置き換えて読み上げます。青空文庫の注記（`［＃…］`）は取り除きます。
- **8GB の制約**: LLM と TTS を同時に載せません。`script`（`run` の前半）の終了後に LLM を解放してから `synth` を行います。
- **文末の句点・読点**: モデルに渡す前に外します（句点・読点で終わると文末に余計な音が入るため）。`script.tsv` の `text` 列は変えません。
- 1 回に合成するのは 40 文字までです。長い文は読点の位置で分けて合成します（長い文は後半で参照音声の声から離れやすいため）。長い文が 30 秒を超えて切り詰められたときは、`warnings.log` に記録されます。

## テスト

```bash
uv sync --extra dev
uv run pytest -q
```

テストは、LLM（Ollama）・TTS（torch / `irodori_tts`）・GPU を使わずに動きます。
