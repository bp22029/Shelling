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


def _slug(name):
    """実験名をファイル名向けに正規化（英数字以外は _ にまとめる）。"""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return s.strip("_")


def figure_path(experiment, kind):
    """可視化PNGの保存パス。kind は 'grid' / 'metrics' など。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{_slug(experiment)}_{kind}.png"


def _results_path(experiment, suffix):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{_slug(experiment)}_{suffix}.csv"


def save_metrics_csv(logs, experiment):
    """各ステップのメトリクスをCSVに保存する。"""
    path = _results_path(experiment, "metrics")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "unsatisfied", "moves", "avg_same_ratio"])
        for l in logs:
            writer.writerow([l.step, l.unsatisfied, l.moves,
                             f"{l.avg_same_ratio:.6f}"])
    print(f"[saved] {path}")
    return path


def save_calibration_csv(results, mode):
    """calibrate_llm の結果（pref -> [(same, diff, ratio, decision), ...]）を保存。"""
    path = _results_path(f"calibration_{mode}", "calibration")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["preference", "same", "diff", "ratio", "decision"])
        for pref, rows in results.items():
            for same, diff, ratio, decision in rows:
                writer.writerow([pref, same, diff, f"{ratio:.6f}", decision])
    print(f"[saved] {path}")
    return path


def save_trials_csv(finals, experiment):
    """run_trials の最終分居度リストを試行番号付きで保存。"""
    path = _results_path(experiment, "trials")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "final_avg_same_ratio"])
        for i, v in enumerate(finals, start=1):
            writer.writerow([i, f"{v:.6f}"])
    print(f"[saved] {path}")
    return path
