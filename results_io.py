"""実験結果の保存先（figures/・results/）とファイル名を一元管理。

- figures/ : 可視化画像（PNG）
- results/ : 数値結果（CSV）

ファイル名は実験名ベースで決まる（再実行すると上書き＝常に最新が残る）。
出力ディレクトリは保存時に自動作成する。
"""

import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
PARSE_FAIL_LOG = RESULTS_DIR / "llm_parse_failures.log"


def _slug(name):
    """実験名をファイル名向けに正規化（英数字以外は _ にまとめる）。"""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def _stem(experiment, model=None):
    """実験名（＋モデル名）からファイル名の語幹を作る。

    model を渡すと experiment__model の形にし、モデルを変えて再実行しても
    結果が上書きされず並存するようにする。
    """
    name = _slug(experiment)
    if model:
        name = f"{name}__{_slug(model)}"
    return name


def figure_path(experiment, kind, model=None):
    """可視化PNGの保存パス。kind は 'grid' / 'metrics' など。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{_stem(experiment, model)}_{kind}.png"


def _results_path(experiment, suffix, model=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{_stem(experiment, model)}_{suffix}.csv"


def save_metrics_csv(logs, experiment, model=None):
    """各ステップのメトリクスをCSVに保存する。"""
    path = _results_path(experiment, "metrics", model)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "unsatisfied", "moves", "avg_same_ratio"])
        for l in logs:
            writer.writerow([l.step, l.unsatisfied, l.moves,
                             f"{l.avg_same_ratio:.6f}"])
    print(f"[saved] {path}")
    return path


def save_calibration_csv(results, mode, model=None):
    """calibrate_llm の結果（pref -> [(same, diff, ratio, decision), ...]）を保存。"""
    path = _results_path(f"calibration_{mode}", "calibration", model)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["preference", "same", "diff", "ratio", "decision"])
        for pref, rows in results.items():
            for same, diff, ratio, decision in rows:
                writer.writerow([pref, same, diff, f"{ratio:.6f}", decision])
    print(f"[saved] {path}")
    return path


def log_llm_parse_failure(model, mode, prompt, raw):
    """FINAL も最終行も判定できなかった LLM 応答の本文をログに追記する。

    どのモデル・どのモードで、どんなプロンプトにどう応答して判定不能になったかを
    後から確認できるようにする。実行をまたいで追記される（不要になったら削除可）。
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARSE_FAIL_LOG, "a", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"model={model}  mode={mode}\n")
        f.write("--- prompt ---\n")
        f.write(f"{(prompt or '').rstrip()}\n")
        f.write("--- raw response ---\n")
        f.write(f"{(raw or '').rstrip()}\n\n")
    return PARSE_FAIL_LOG


def save_trials_csv(finals, experiment, model=None):
    """run_trials の最終分居度リストを試行番号付きで保存。"""
    path = _results_path(experiment, "trials", model)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "final_avg_same_ratio"])
        for i, v in enumerate(finals, start=1):
            writer.writerow([i, f"{v:.6f}"])
    print(f"[saved] {path}")
    return path
