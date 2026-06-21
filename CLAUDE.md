# CLAUDE.md

このファイルは、このリポジトリで作業する際に Claude Code（および開発者）が従うべきガイドラインをまとめたものです。

## プロジェクト概要

シェリングの分居モデル（Schelling segregation model）の実験コード。エージェントの意思決定を「ルールベース」と「LLMベース」で差し替えながら、分居（segregation）の発生傾向を比較する研究用プログラム。

- エージェントは **タイプA / タイプB**（旧称: チームA/チームB）の2種類。`schelling_core.TYPE_A` / `TYPE_B` で表現する。
- 意思決定器は `decision.DecisionMaker` インターフェース（`decide(agent_type, same, diff, empty, preference)`）で統一。ルール版・LLM版を差し替え可能。
- LLM はラボサーバー（oMLX / OpenAI 互換）に HTTP 接続する想定。

## モジュール構成

| ファイル | 役割 |
|---|---|
| `schelling_core.py` | 格子空間・近傍計算・移動処理・`run_simulation` 本体。LLM に依存しない。 |
| `decision.py` | 意思決定器。`RuleFixedDecision` / `RuleHeterogeneousDecision` / `LLMDecision`。 |
| `experiments.py` | 実験定義（Exp0〜4）・キャリブレーション・複数試行の集計。 |
| `llm_client.py` | LLM クライアント群（`HTTPLLMClient` / `DummyLLMClient` / `LocalLLMClient`）。 |
| `viz.py` | 可視化（初期/最終配置・メトリクス）。 |
| `run.py` | エントリポイント。`python run.py {rule,calib,no_pref,numeric,verbal}`。 |
| `test_connection.py` / `inference_test.py` | サーバー接続・推論の動作確認。 |

## 環境構築（conda）

このプロジェクトは conda 環境（名前: `schelling`）を使う。定義は `environment.yml`。

```bash
conda env create -f environment.yml   # 初回作成
conda activate schelling              # 有効化
conda env update -f environment.yml --prune   # 依存を変えたとき
```

依存パッケージを追加・変更したら `environment.yml` を更新すること
（`requirements.txt` は pip 利用者向けに残してあるので、合わせて更新する）。

VS Code では `Ctrl+Shift+P` →「Python: Select Interpreter」で conda 環境
`schelling` を選ぶと、▶ ボタン実行もこの環境を使う。

## 実行方法

### VS Code でボタン起動（推奨・引数入力不要）

「実行とデバッグ」パネル（`Ctrl+Shift+D`）左上のドロップダウンで実験を選び、緑の ▶ ボタン（または `F5`）を押すだけ。構成は `.vscode/launch.json` に定義済み。

- `Exp0+4: Rule (LLM不要)`
- `Calibration (verbal / 要サーバー)`
- `Exp1: LLM No-Preference (要サーバー)`
- `Exp2: LLM Numeric (要サーバー)`
- `Exp3: LLM Verbal (要サーバー)`

新しい実験を追加したら `.vscode/launch.json` にも構成を1つ追加すること。

### コマンドライン

```bash
# LLM 不要（サーバー無しで動作確認できる）
python run.py rule

# 以下は LLM サーバーが必要
python run.py calib      # verbal 選好のキャリブレーション
python run.py no_pref    # Exp1
python run.py numeric    # Exp2
python run.py verbal     # Exp3
```

依存パッケージは `requirements.txt`（`pip install -r requirements.txt`）。

## 命名規約

- エージェントの所属は **「タイプ（type）」** で統一する。「チーム / team」という語は使わない。
  - 定数: `TYPE_A` / `TYPE_B`
  - 変数・引数: `agent_type`（Python 組み込みの `type` と衝突させないため `type` 単体は使わない）
  - プロンプト内の表示文字列は「Aタイプ / Bタイプ」。

## Git バージョン管理ルール（必須）

このプロジェクトは研究用のため、**`main` には常に安定版（動作確認済みのコード）があること**を最優先とする。
ブランチ構成は Git Flow に倣いつつ、ソロ研究運用に合わせて **`main` + `feature` の2層**とする（`develop` は使わない）。

### 基本原則

- **`main` に直接コミットしない。** どんなに小さな修正でも必ず `feature` ブランチを切ってから作業する。
- 1つの機能追加・修正につき1つの `feature` ブランチ。
- 作業が完了し動作確認が済んだら `main` にマージし、その `feature` ブランチは削除する。

### ブランチ命名

- `feature/<簡潔な内容>` 形式。例: `feature/rename-team-to-type`, `feature/add-exp5`, `feature/fix-neighbor-edge`。

### 作業フロー

```bash
# 1. main を最新にして feature ブランチを切る
git switch main
git pull
git switch -c feature/<内容>

# 2. コードを変更・コミット（feature ブランチ上で作業）
git add -A
git commit -m "<変更内容>"

# 3. 動作確認（最低限 LLM 不要のルール版が通ること）
python run.py rule

# 4. main にマージ
git switch main
git merge --no-ff feature/<内容>

# 5. マージ済みの feature ブランチを削除
git branch -d feature/<内容>
```

### 注意点

- マージは `--no-ff` を推奨（feature 単位の履歴が残り、後から追いやすい）。
- `main` にマージする前に、必ずそのブランチ上で動作確認を済ませること。壊れたコードを `main` に入れない。
- コミット／マージ／プッシュは、ユーザーから依頼があったときに行う（勝手にプッシュしない）。
- `.env`（秘密情報）は `.gitignore` 済み。コミットに含めない。
