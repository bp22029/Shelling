"""LLM の応答と判定を確認するデバッグ用スクリプト。

同種隣人数を 0〜8 まで振り、各ケースで
「判定結果」と「応答の最終行（結論の書き方）」を一覧表示する。
判定が同種率に対して素直に動いているか（stay/move の偏りがないか）を確認できる。

使い方:
    python debug_raw.py            # numeric モード
    python debug_raw.py verbal     # mode 指定（no_pref / numeric / verbal）
    python debug_raw.py numeric 4  # 末尾に same を渡すと、そのケースだけ生応答を全文表示
"""
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from llm_client import HTTPLLMClient
from decision import LLMDecision
from schelling_core import TYPE_A, PREF_MID

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

mode = sys.argv[1] if len(sys.argv) > 1 else "numeric"
focus = int(sys.argv[2]) if len(sys.argv) > 2 else None  # この same だけ全文表示

client = HTTPLLMClient()  # thinking は既定でオン
dm = LLMDecision(client, mode=mode)


def parse(raw):
    """decision.py と同じ判定（FINAL マーカー優先 → 最終行 → 不能）。"""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).lower()
    marks = re.findall(r"final\s*[:：]\s*(stay|move)", text)
    if marks:
        return marks[-1], "FINAL"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if "stay" in last and "move" not in last:
            return "stay", "最終行"
        if "move" in last and "stay" not in last:
            return "move", "最終行"
    return None, "判定不能"


def last_line(raw):
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


print(f"===== mode={mode} / numeric基準は同種30%以上でstay =====")
print(f"{'same':>4} {'diff':>4} {'ratio':>6} {'期待':>5} {'判定':>5} {'抽出元':>6}  最終行")
for same in range(0, 9):
    diff = 8 - same
    ratio = same / 8
    expected = "stay" if ratio >= 0.30 else "move"
    prompt = dm.build_prompt(agent_type=TYPE_A, same=same, diff=diff,
                             empty=0, preference=PREF_MID)
    raw = client.generate(prompt)
    decision, source = parse(raw)
    flag = "" if decision == expected else "  <== 不一致"
    print(f"{same:>4} {diff:>4} {ratio:>6.2f} {expected:>5} {str(decision):>5} {source:>6}  "
          f"{last_line(raw)[:50]}{flag}")
    if focus is not None and same == focus:
        print("\n----- 生応答（全文）-----")
        print(raw)
        print(f"----- 抽出: {decision}（{source}）-----\n")
