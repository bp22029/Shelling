"""LLM単体の推論時間計測 ＋ <think>/<answer> タグ回収の確認。

シミュレーションは一切含まない。研究室サーバー(oMLX)に1問ずつ投げて、
  - 1回あたりの推論時間（複数回の平均/最小/最大）
  - 応答テキスト(raw)
  - <think>...</think> と <answer>...</answer> を本文から抽出できるか
  - 思考が「別フィールド(reasoning_content)」で返っていないか
  - トークン使用量（取得できれば）
を表示する。

使い方（schelling フォルダ内・.env が必要）:
    python inference_test.py        # 既定プロンプトで3回計測
    python inference_test.py 5      # 5回計測
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ----- ここを自分の本番プロンプトに差し替えて試せる -----
# 論文の出力形式（<think>/<answer>）に準拠した指示。
SYSTEM_PROMPT = """\
あなたは簡潔に回答するアシスタントです。

# 出力形式（重要）
回答は必ず以下の形式で出力してください。
<think>
ここに思考過程を書く
</think>
<answer>
ここに最終的な答えだけを書く
</answer>"""

USER_PROMPT = "日本で一番高い山は何ですか。"
# --------------------------------------------------------

# thinking を有効にしたいので /no_think は付けない（タグ回収の検証が目的）。
TEMPERATURE = 0.7
MAX_TOKENS = 1024  # thinking 分の余裕

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract(tag_re, text):
    m = tag_re.search(text)
    return m.group(1).strip() if m else None


def get_reasoning_field(message):
    """サーバーが思考を本文ではなく別フィールドで返す場合に拾う（無ければNone）。"""
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val:
            return val
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        if extra.get(key):
            return extra[key]
    return None


def main():
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )
    model = os.environ["LLM_MODEL"]

    print(f"model = {model}")
    print(f"runs  = {n}")
    print("=" * 60)

    times = []
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning_field = get_reasoning_field(msg)

        think = extract(THINK_RE, content)
        answer = extract(ANSWER_RE, content)

        print(f"\n----- run {i}: {elapsed:.2f}s -----")
        print(f"finish_reason = {resp.choices[0].finish_reason}")
        if resp.usage:
            print(f"tokens: prompt={resp.usage.prompt_tokens}, "
                  f"completion={resp.usage.completion_tokens}, "
                  f"total={resp.usage.total_tokens}")
        print(f"<think>  本文から抽出: {'OK' if think is not None else 'NG（本文に無し）'}")
        print(f"<answer> 本文から抽出: {'OK' if answer is not None else 'NG（本文に無し）'}")
        if reasoning_field is not None:
            print("※ 思考は本文ではなく別フィールド(reasoning_content等)で返っています")

        print("--- raw content ---")
        print(content)
        if reasoning_field is not None:
            print("--- reasoning field ---")
            print(reasoning_field)
        if think is not None:
            print("--- think (本文から) ---")
            print(think)
        if answer is not None:
            print("--- answer (本文から) ---")
            print(answer)

    print("\n" + "=" * 60)
    avg = sum(times) / len(times)
    print(f"time: avg={avg:.2f}s, min={min(times):.2f}s, max={max(times):.2f}s "
          f"(n={len(times)})")


if __name__ == "__main__":
    main()
