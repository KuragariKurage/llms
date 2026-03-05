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
- [fzf](https://github.com/junegunn/fzf)
- [uv](https://github.com/astral-sh/uv)

### クイックインストール

```bash
./install.sh
```

### 手動インストール

```bash
uv tool install -e .
```

これで `llms` コマンドがどこからでも使えるようになります。

## 使い方

```bash
llms                  # fzf で検索 → Enter で model ID をクリップボードにコピー
llms --refresh        # キャッシュを強制更新
llms --no-copy        # コピーせず stdout に出力（パイプ用）
llms --json           # 選択したモデルの詳細を JSON で出力
llms -p anthropic     # プロバイダでフィルタ
```

### キー操作

| キー | 動作 |
|------|------|
| 文字入力 | fuzzy search |
| 上下キー | モデル選択 |
| Enter | model ID をクリップボードにコピー |
| Ctrl-C | 終了 |

## データソース

[models.dev](https://models.dev/) の `/api.json` エンドポイントからモデル情報を取得します。
データは `~/Library/Caches/llms/` に1時間キャッシュされます。

## 開発

```bash
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
```
