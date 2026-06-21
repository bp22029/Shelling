"""可視化。cell 7 の移植。

plot_grid / plot_initial_final / plot_metrics はスクリプト実行でもウィンドウ表示できる。
animate_logs はノートブック表示用（IPythonを遅延importするので、スクリプトでimportしても壊れない）。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation


def plot_grid(grid_array, title="Schelling", ax=None):
    cmap = mcolors.ListedColormap(["white", "#e74c3c", "#3498db"])  # 空・赤A・青B
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    return ax


def plot_initial_final(logs, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_grid(logs[0].grid_snapshot, f"{title_prefix} Step 0", ax=axes[0])
    plot_grid(logs[-1].grid_snapshot, f"{title_prefix} Step {logs[-1].step}", ax=axes[1])
    plt.tight_layout()
    plt.show()


def plot_metrics(logs):
    steps = [l.step for l in logs]
    unsat = [l.unsatisfied for l in logs]
    ratio = [l.avg_same_ratio for l in logs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, unsat, marker="o")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("unsatisfied agents")
    axes[0].set_title("Unsatisfied over time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, ratio, marker="o", color="green")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("avg same-neighbor ratio")
    axes[1].set_title("Segregation degree")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def animate_logs(logs, interval=300):
    """全ステップのアニメーション。ノートブックでは表示用HTMLを返す。"""
    from IPython.display import HTML  # ノートブック以外では未使用なので遅延import

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = mcolors.ListedColormap(["white", "#e74c3c", "#3498db"])

    def update(i):
        ax.clear()
        ax.imshow(logs[i].grid_snapshot, cmap=cmap, vmin=0, vmax=2)
        ax.set_title(f"Step {logs[i].step} (avg_same={logs[i].avg_same_ratio:.3f})")
        ax.set_xticks([])
        ax.set_yticks([])

    anim = FuncAnimation(fig, update, frames=len(logs), interval=interval)
    plt.close(fig)
    return HTML(anim.to_jshtml())
