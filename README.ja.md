# llms

[models.dev](https://models.dev/) のLLMモデルカタログをCLIからfuzzy searchできるツール。

## デモ

```
$ llms
Model> claude-sonnet-4.6

  anthropic/claude-sonnet-4-6          | Claude Sonnet 4.6
  opencode/claude-sonnet-4-6           |
  venice/claude-sonnet-4-6             | Provider:    Anthropic
  ...                                  | Model ID:    anthropic/claude-sonnet-4-6
                                       | Cost:        $3.00 / $15.00
                                       | Context:     200K

Copied: anthropic/claude-sonnet-4-6
```

## インストール

### 前提条件

- Python 3.13+
- [fzf](https://github.com/junegunn/fzf)（インタラクティブモード用）

### インストール

```bash
# uv（推奨）
uv tool install -e .

# pipx
pipx install -e .

# pip
pip install -e .
```

または `./install.sh` を実行すると、利用可能なインストーラを自動検出します。

#### GitHub から直接インストール（clone 不要）

```bash
uv tool install git+https://github.com/atsuya.sakata/llm-models.git
pipx install git+https://github.com/atsuya.sakata/llm-models.git
```

これで `llms` コマンドがどこからでも使えるようになります。

## 使い方

### インタラクティブモード（fzf）

```bash
llms                  # fzf で検索 → Enter で model ID をクリップボードにコピー
llms --refresh        # キャッシュを強制更新
llms --no-copy        # コピーせず stdout に出力（パイプ用）
llms --json           # 選択したモデルの詳細を JSON で出力
llms -p anthropic     # プロバイダでフィルタ
```

#### キー操作

| キー | 動作 |
|------|------|
| 文字入力 | fuzzy search |
| 上下キー | モデル選択 |
| Enter | model ID をクリップボードにコピー |
| Ctrl-C | 終了 |

### プログラマティックモード（AI エージェント・スクリプト向け）

fzf 不要の非インタラクティブサブコマンド。構造化データを出力します。

```bash
# ID 指定でモデル取得
llms get anthropic/claude-sonnet-4-6 --json

# フィルタ付きモデル一覧
llms list --json
llms list --cap tool_call --min-context 128k --sort cost.input --limit 5 --json
llms list --cap reasoning --max-input-cost 5.0 --json

# テキスト検索
llms search claude --json --limit 5
llms search llama -p meta --json

# プロバイダ一覧
llms providers --json
```

#### フィルタフラグ（`list` / `search` 共通）

| フラグ | 説明 | 例 |
|--------|------|-----|
| `-p`, `--provider` | プロバイダで絞り込み | `-p anthropic` |
| `--cap` | ケイパビリティ（複数指定可、AND条件） | `--cap tool_call --cap reasoning` |
| `--min-context` | 最小コンテキストウィンドウ | `--min-context 128k` |
| `--max-input-cost` | 最大入力コスト（$/1Mトークン） | `--max-input-cost 3.0` |
| `--sort` | ソートフィールド | `--sort cost.input` |
| `--limit` | 最大件数 | `--limit 10` |

#### 出力形式

| フラグ | 形式 |
|--------|------|
| `--json` | 整形済み JSON |
| `--jsonl` | 1行1 JSON オブジェクト |
| *(なし)* | 1行1モデル ID |

### Python ライブラリ

```python
from llms.client import Client
from llms.query import Query

client = Client()

# ID 指定で取得
model = client.get("anthropic/claude-sonnet-4-6")

# フィルタ付き一覧
models = client.list(Query(
    caps=["tool_call"],
    min_context=128_000,
    sort="cost.input",
    limit=5,
))

# テキスト検索
results = client.search("claude")

# プロバイダ一覧
providers = client.providers()
```

## データソース

[models.dev](https://models.dev/) の `/api.json` エンドポイントからモデル情報を取得します。
データは `~/Library/Caches/llms/` に1時間キャッシュされます。

## 開発

```bash
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
```
