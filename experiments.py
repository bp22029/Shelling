"""実験定義。cell 8〜9 の移植 ＋ 複数試行の集計ヘルパ。

- calibrate_llm: シミュレーション前に、LLMが各選好で同種割合にどう答えるか確認
- run_rule_fixed_demo / run_rule_hetero_demo: LLM不要のルール版（Exp0 / Exp4）
- run_llm_demo: LLM版（Exp1=no_pref / Exp2=numeric / Exp3=verbal）
- run_trials: 同一条件をN回実行し、最終分居度の分布を返す（温度>0前提の傾向確認用）
"""

import time

from schelling_core import (
    SchellingGrid, run_simulation, TYPE_A, PREF_LOW, PREF_MID, PREF_HIGH,
)
from decision import (
    RuleFixedDecision, RuleHeterogeneousDecision, LLMDecision,
)
from viz import plot_initial_final, plot_metrics
from results_io import (
    save_metrics_csv, save_calibration_csv, save_trials_csv, figure_path,
)


LLM_LABELS = {
    "no_pref": "Experiment 1: LLM-NoPreference",
    "numeric": "Experiment 2: LLM-NumericPreference",
    "verbal":  "Experiment 3: LLM-VerbalPreference",
}


def model_label(llm_client):
    """LLMクライアントから表示用のモデル名を取り出す（無ければクラス名）。"""
    return (getattr(llm_client, "model", None)
            or getattr(llm_client, "model_name", None)
            or type(llm_client).__name__)


def _emit_outputs(logs, label, title_prefix, show=True, save=True, model=None):
    """数値CSV（results/）と可視化PNG（figures/）を保存しつつ、必要なら表示する。

    model を渡すと、ファイル名（モデル別保存）と図のキャプションにモデル名を含める。
    """
    caption = f"model: {model}" if model else None
    if save:
        save_metrics_csv(logs, label, model=model)
    if show or save:
        plot_initial_final(logs, title_prefix,
                           save_path=figure_path(label, "grid", model) if save else None,
                           show=show, caption=caption)
        plot_metrics(logs,
                     save_path=figure_path(label, "metrics", model) if save else None,
                     show=show, caption=caption)


def calibrate_llm(llm_client, mode="verbal", save=True):
    """周囲構成を変えながらLLMの判断を確認。"""
    decision_maker = LLMDecision(llm_client, mode=mode)
    model = model_label(llm_client)
    results = {}
    prefs_to_test = [PREF_LOW, PREF_MID, PREF_HIGH] if mode == "verbal" else [PREF_MID]

    print(f"=== LLM Calibration (mode={mode}) ===")
    print(f"接続モデル: {model}", flush=True)
    for pref in prefs_to_test:
        print(f"\n[preference: {pref}]")
        print(f"{'same':>5} {'diff':>5} {'ratio':>7} -> {'decision':>10}")
        results[pref] = []
        for same in range(0, 9):
            diff = 8 - same
            total = same + diff
            ratio = same / total if total > 0 else 0
            decision = decision_maker.decide(
                agent_type=TYPE_A, same=same, diff=diff, empty=0, preference=pref
            )
            results[pref].append((same, diff, ratio, decision))
            print(f"{same:>5} {diff:>5} {ratio:>7.3f} -> {decision:>10}")
    if save:
        save_calibration_csv(results, mode, model=model)
    return results


def run_rule_fixed_demo(size=10, max_steps=30, seed=42, show=False, save=True):
    label = "Experiment 0: Rule-Fixed"
    print(f"\n========= {label} =========")
    grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=seed)
    decision = RuleFixedDecision(similar_wanted=0.30)
    logs = run_simulation(grid, decision, max_steps=max_steps, seed=seed)
    _emit_outputs(logs, label, "Rule-Fixed", show=show, save=save)
    return logs


def run_rule_hetero_demo(size=10, max_steps=30, seed=42, show=False, save=True):
    label = "Experiment 4: Rule-Heterogeneous"
    print(f"\n========= {label} =========")
    grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=seed)
    decision = RuleHeterogeneousDecision(low=0.25, mid=0.40, high=0.55)
    logs = run_simulation(grid, decision, max_steps=max_steps, seed=seed)
    _emit_outputs(logs, label, "Rule-Heterogeneous", show=show, save=save)
    return logs


def run_llm_demo(llm_client, mode="verbal", size=10, max_steps=10, seed=42,
                 show=False, save=True, progress_every=10):
    label = LLM_LABELS[mode]
    model = model_label(llm_client)
    print(f"\n========= {label} =========")
    print(f"接続モデル: {model}", flush=True)
    grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=seed)
    n_agents = len(grid.agent_positions())
    print(f"格子 {size}x{size} / エージェント {n_agents}体 / max_steps={max_steps}")
    print(f"→ LLM呼び出しは最大 約 {n_agents * max_steps} 回。"
          f"途中経過は {progress_every} 体ごとに表示します。", flush=True)
    decision = LLMDecision(llm_client, mode=mode)
    t0 = time.time()
    logs = run_simulation(grid, decision, max_steps=max_steps, seed=seed,
                          progress_every=progress_every)
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"LLM calls: {decision.call_count}, errors: {decision.error_count}")
    _emit_outputs(logs, label, label, show=show, save=save, model=model)
    return logs


def run_trials(make_decision, n_trials=5, size=10, max_steps=10,
               grid_seed=42, sim_seed=42, save=True, label="trials"):
    """同一条件をN回実行し、最終avg_same_ratioのリストと全logsを返す。

    grid_seed / sim_seed を固定すれば、試行間のばらつきはLLMの確率性（temperature>0）に由来する。
    make_decision は毎回新しい意思決定器を返すファクトリ（例: lambda: LLMDecision(client, "verbal")）。
    """
    finals, all_logs = [], []
    for t in range(n_trials):
        grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=grid_seed)
        decision = make_decision()
        logs = run_simulation(grid, decision, max_steps=max_steps,
                              seed=sim_seed, verbose=False)
        finals.append(logs[-1].avg_same_ratio)
        all_logs.append(logs)
        print(f"trial {t+1}/{n_trials}: final avg_same_ratio = {finals[-1]:.3f}")
    if finals:
        import numpy as np
        print(f"--- mean={np.mean(finals):.3f}, std={np.std(finals):.3f} (n={n_trials})")
    if save and finals:
        save_trials_csv(finals, label)
    return finals, all_logs
