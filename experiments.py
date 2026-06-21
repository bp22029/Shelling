"""実験定義。cell 8〜9 の移植 ＋ 複数試行の集計ヘルパ。

- calibrate_llm: シミュレーション前に、LLMが各選好で同種割合にどう答えるか確認
- run_rule_fixed_demo / run_rule_hetero_demo: LLM不要のルール版（Exp0 / Exp4）
- run_llm_demo: LLM版（Exp1=no_pref / Exp2=numeric / Exp3=verbal）
- run_trials: 同一条件をN回実行し、最終分居度の分布を返す（温度>0前提の傾向確認用）
"""

import time

from schelling_core import (
    SchellingGrid, run_simulation, TEAM_A, PREF_LOW, PREF_MID, PREF_HIGH,
)
from decision import (
    RuleFixedDecision, RuleHeterogeneousDecision, LLMDecision,
)
from viz import plot_initial_final, plot_metrics


LLM_LABELS = {
    "no_pref": "Experiment 1: LLM-NoPreference",
    "numeric": "Experiment 2: LLM-NumericPreference",
    "verbal":  "Experiment 3: LLM-VerbalPreference",
}


def calibrate_llm(llm_client, mode="verbal"):
    """周囲構成を変えながらLLMの判断を確認。"""
    decision_maker = LLMDecision(llm_client, mode=mode)
    results = {}
    prefs_to_test = [PREF_LOW, PREF_MID, PREF_HIGH] if mode == "verbal" else [PREF_MID]

    print(f"=== LLM Calibration (mode={mode}) ===")
    for pref in prefs_to_test:
        print(f"\n[preference: {pref}]")
        print(f"{'same':>5} {'diff':>5} {'ratio':>7} -> {'decision':>10}")
        results[pref] = []
        for same in range(0, 9):
            diff = 8 - same
            total = same + diff
            ratio = same / total if total > 0 else 0
            decision = decision_maker.decide(
                team=TEAM_A, same=same, diff=diff, empty=0, preference=pref
            )
            results[pref].append((same, diff, ratio, decision))
            print(f"{same:>5} {diff:>5} {ratio:>7.3f} -> {decision:>10}")
    return results


def run_rule_fixed_demo(size=10, max_steps=30, seed=42, show=True):
    print("\n========= Experiment 0: Rule-Fixed =========")
    grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=seed)
    decision = RuleFixedDecision(similar_wanted=0.30)
    logs = run_simulation(grid, decision, max_steps=max_steps, seed=seed)
    if show:
        plot_initial_final(logs, "Rule-Fixed")
        plot_metrics(logs)
    return logs


def run_rule_hetero_demo(size=10, max_steps=30, seed=42, show=True):
    print("\n========= Experiment 4: Rule-Heterogeneous =========")
    grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=seed)
    decision = RuleHeterogeneousDecision(low=0.25, mid=0.40, high=0.55)
    logs = run_simulation(grid, decision, max_steps=max_steps, seed=seed)
    if show:
        plot_initial_final(logs, "Rule-Heterogeneous")
        plot_metrics(logs)
    return logs


def run_llm_demo(llm_client, mode="verbal", size=10, max_steps=10, seed=42, show=True):
    label = LLM_LABELS[mode]
    print(f"\n========= {label} =========")
    grid = SchellingGrid(size=size, empty_rate=0.2, ratio_a=0.5, seed=seed)
    decision = LLMDecision(llm_client, mode=mode)
    t0 = time.time()
    logs = run_simulation(grid, decision, max_steps=max_steps, seed=seed)
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"LLM calls: {decision.call_count}, errors: {decision.error_count}")
    if show:
        plot_initial_final(logs, label)
        plot_metrics(logs)
    return logs


def run_trials(make_decision, n_trials=5, size=10, max_steps=10,
               grid_seed=42, sim_seed=42):
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
    return finals, all_logs
