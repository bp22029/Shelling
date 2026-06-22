"""空間管理・近傍計算・移動処理・シミュレーション本体。LLM部分とは独立。

cell 3（SchellingGrid等）と cell 6（run_simulation）を統合したモジュール。
"""

import random
import time
from dataclasses import dataclass
import numpy as np

EMPTY = 0
TYPE_A = 1
TYPE_B = 2

# 選好タイプ（LLM-Verbal と Rule-Hetero で共通の割り当てに使う）
PREF_LOW = "low"
PREF_MID = "mid"
PREF_HIGH = "high"


@dataclass
class Agent:
    """1エージェントの状態。位置は格子側で管理するのでタイプと選好のみ。"""
    agent_type: int
    preference: str = PREF_MID


class SchellingGrid:
    """格子空間とエージェント配置・移動を管理。LLMには依存しない。"""

    def __init__(self, size=10, empty_rate=0.2, ratio_a=0.5, seed=42):
        self.size = size
        self.seed = seed
        rng = random.Random(seed)

        total = size * size
        n_empty = int(total * empty_rate)
        n_agent = total - n_empty
        n_a = int(n_agent * ratio_a)
        n_b = n_agent - n_a

        cells = [TYPE_A] * n_a + [TYPE_B] * n_b + [EMPTY] * n_empty
        rng.shuffle(cells)
        self.grid = np.array(cells).reshape(size, size)

        # 選好タイプを低/中/高で1/3ずつ割り当て（シード固定）
        self.preferences = {}
        positions = [(r, c) for r in range(size) for c in range(size)
                     if self.grid[r, c] != EMPTY]
        rng.shuffle(positions)
        n_each = len(positions) // 3
        for i, pos in enumerate(positions):
            if i < n_each:
                self.preferences[pos] = PREF_LOW
            elif i < 2 * n_each:
                self.preferences[pos] = PREF_MID
            else:
                self.preferences[pos] = PREF_HIGH

    def neighbors(self, r, c):
        """Moore近傍（周囲8マス）を返す。端は折り返しなし。"""
        same, diff, empty = 0, 0, 0
        my_type = self.grid[r, c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    cell = self.grid[nr, nc]
                    if cell == EMPTY:
                        empty += 1
                    elif cell == my_type:
                        same += 1
                    else:
                        diff += 1
        return same, diff, empty

    def empty_cells(self):
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.grid[r, c] == EMPTY]

    def agent_positions(self):
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.grid[r, c] != EMPTY]

    def move(self, from_pos, to_pos, rng):
        """from_pos のエージェントを to_pos（空き地）に移動。選好も一緒に移す。"""
        r1, c1 = from_pos
        r2, c2 = to_pos
        self.grid[r2, c2] = self.grid[r1, c1]
        self.grid[r1, c1] = EMPTY
        self.preferences[(r2, c2)] = self.preferences.pop(from_pos)

    def avg_same_ratio(self):
        """全エージェントの「近隣同種率」平均。分居度の指標。"""
        ratios = []
        for r, c in self.agent_positions():
            same, diff, _ = self.neighbors(r, c)
            total = same + diff
            if total > 0:
                ratios.append(same / total)
        return np.mean(ratios) if ratios else 0.0


@dataclass
class StepLog:
    step: int
    unsatisfied: int
    moves: int
    avg_same_ratio: float
    grid_snapshot: np.ndarray


def run_simulation(grid, decision_maker, max_steps=30, seed=42, verbose=True,
                   progress_every=0):
    """シミュレーション本体。1ステップごとに不満足者を空き地へランダム移動。

    【更新方式：逐次（非同期）更新。意図的な設計なので変更しないこと】
    エージェントを1体ずつ処理し、各エージェントは grid.neighbors() で
    「それまでの移動が反映済みの最新の盤面」を見て判断し、その場で移動する。
    これはシェリングが盤上のコマを1つずつ動かして見せた逐次的な再配置に対応し、
    「ある個体の移動が後続の判断に波及する」という本モデルの本質的な性質を保つ。
    ※ ステップ内でスナップショットを固定して一斉判断する「同期更新」にすると
      この波及が消え、別ダイナミクスのモデルになる。1ステップ内の判断は互いに
      依存するため、忠実性を保つ限りステップ内を並列化することはできない
      （高速化したいときは、独立したラン／試行／条件の同時実行で並列度を稼ぐ）。

    grid初期配置・移動先選択は seed 固定で決定的。LLMの揺らぎは
    decision_maker（temperature>0）側から入る設計。

    progress_every>0 のとき、1ステップ内で progress_every 体処理するごとに
    途中経過を出力する（LLMのように1件ずつ時間がかかる場合の生存確認用）。
    """
    rng = random.Random(seed)
    logs = [StepLog(step=0, unsatisfied=0, moves=0,
                    avg_same_ratio=grid.avg_same_ratio(),
                    grid_snapshot=grid.grid.copy())]

    for step in range(1, max_steps + 1):
        positions = grid.agent_positions()
        rng.shuffle(positions)
        n_total = len(positions)
        t_step = time.time()

        unsatisfied = 0
        moves = 0
        for i, pos in enumerate(positions, start=1):
            r, c = pos
            if grid.grid[r, c] == EMPTY:  # この回で空になっている場合
                continue
            agent_type = grid.grid[r, c]
            same, diff, empty = grid.neighbors(r, c)
            pref = grid.preferences.get(pos, PREF_MID)

            decision = decision_maker.decide(agent_type, same, diff, empty, pref)
            if decision == "move":
                unsatisfied += 1
                empties = grid.empty_cells()
                if empties:
                    target = rng.choice(empties)
                    grid.move(pos, target, rng)
                    moves += 1

            # 1ステップ内の途中経過（LLM実行が遅いときの進捗確認用）
            if progress_every and (i % progress_every == 0 or i == n_total):
                errs = getattr(decision_maker, "error_count", None)
                extra = f", errors={errs}" if errs is not None else ""
                print(f"  step {step:3d}: {i:3d}/{n_total} 体処理 | "
                      f"unsatisfied={unsatisfied}, moves={moves}{extra}",
                      flush=True)

        ratio = grid.avg_same_ratio()
        logs.append(StepLog(step=step, unsatisfied=unsatisfied, moves=moves,
                            avg_same_ratio=ratio, grid_snapshot=grid.grid.copy()))
        if verbose:
            dt = time.time() - t_step
            print(f"Step {step:3d}: unsatisfied={unsatisfied:3d}, "
                  f"moves={moves:3d}, avg_same={ratio:.3f} ({dt:.1f}s)", flush=True)
        if unsatisfied == 0:
            if verbose:
                print(f"Converged at step {step}.", flush=True)
            break

    return logs
