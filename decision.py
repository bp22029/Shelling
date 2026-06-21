"""意思決定器。ルール版とLLM版で差し替わる部分。インターフェースを揃える。

cell 4 の移植。元ノートでは LLMDecision.decide がクラス外に出てしまっていた
（＝抽象クラスのままインスタンス化不可）バグを修正し、クラス内に配置した。
"""

import re
from abc import ABC, abstractmethod

from schelling_core import TYPE_A, PREF_LOW, PREF_MID, PREF_HIGH


class DecisionMaker(ABC):
    @abstractmethod
    def decide(self, agent_type: int, same: int, diff: int, empty: int,
               preference: str) -> str:
        """'stay' または 'move' を返す。"""
        ...


class RuleFixedDecision(DecisionMaker):
    """実験0: 全員が同じ閾値。"""

    def __init__(self, similar_wanted=0.30):
        self.threshold = similar_wanted

    def decide(self, agent_type, same, diff, empty, preference):
        total = same + diff
        if total == 0:
            return "stay"
        ratio = same / total
        return "stay" if ratio >= self.threshold else "move"


class RuleHeterogeneousDecision(DecisionMaker):
    """実験4: 選好タイプごとに閾値を変える。"""

    def __init__(self, low=0.25, mid=0.40, high=0.55):
        self.thresholds = {PREF_LOW: low, PREF_MID: mid, PREF_HIGH: high}

    def decide(self, agent_type, same, diff, empty, preference):
        total = same + diff
        if total == 0:
            return "stay"
        ratio = same / total
        return "stay" if ratio >= self.thresholds[preference] else "move"


class LLMDecision(DecisionMaker):
    """実験1〜3: LLMで判定。mode で プロンプトの種類を指定。"""

    def __init__(self, llm_client, mode):  # mode: "no_pref" / "numeric" / "verbal"
        self.llm = llm_client
        self.mode = mode
        self.call_count = 0
        self.error_count = 0

    def build_prompt(self, agent_type, same, diff, empty, preference):
        type_name = "A" if agent_type == TYPE_A else "B"
        base_situation = (
            f"あなたの周囲には以下の住民がいます。\n"
            f"- Aタイプ: {same if agent_type == TYPE_A else diff}人\n"
            f"- Bタイプ: {diff if agent_type == TYPE_A else same}人\n"
            f"- 空き地: {empty}つ\n\n"
            f"あなたは現在の場所に住み続けますか。\n"
            f"それとも移動しますか。\n\n"
            f"必ず stay または move のどちらかだけで答えてください。"
        )

        if self.mode == "no_pref":
            return f"あなたは{type_name}タイプの住民です。\n\n{base_situation}"
        elif self.mode == "numeric":
            return (
                f"あなたは{type_name}タイプの住民です。\n\n"
                f"あなたは、周囲の住民のうち30%以上が{type_name}タイプであれば満足します。\n"
                f"30%未満であれば移動したいと考えます。\n\n"
                f"{base_situation}"
            )
        elif self.mode == "verbal":
            pref_text = {
                PREF_LOW: (
                    "あなたは、自分と異なるタイプの住民が近くにいてもあまり気にしません。\n"
                    "近隣が多様であっても、基本的には住み続ける傾向があります。"
                ),
                PREF_MID: (
                    "あなたは、近隣にある程度は自分と同じタイプの住民がいてほしいと考えています。\n"
                    "ただし、異なるタイプの住民が多少いても大きな問題とは考えません。"
                ),
                PREF_HIGH: (
                    "あなたは、近隣に自分と同じタイプの住民が多いことを強く望みます。\n"
                    "異なるタイプの住民が多い環境では、移動したいと感じやすいです。"
                ),
            }[preference]
            return f"あなたは{type_name}タイプの住民です。\n\n{pref_text}\n\n{base_situation}"
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def decide(self, agent_type, same, diff, empty, preference):
        prompt = self.build_prompt(agent_type, same, diff, empty, preference)
        self.call_count += 1
        raw_output = self.llm.generate(prompt)

        # thinking ブロックを除去してから判定（thinkingオン/オフどちらでも安全）
        answer_part = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
        answer_part = answer_part.strip().lower()

        if "stay" in answer_part and "move" not in answer_part:
            return "stay"
        elif "move" in answer_part and "stay" not in answer_part:
            return "move"
        else:
            self.error_count += 1
            return "stay"
