"""エントリポイント。

使い方:
    python run.py rule       # Exp0 + Exp4（LLM不要・サーバー無しで動作確認できる）
    python run.py calib      # verbal選好のキャリブレーション（要サーバー）
    python run.py no_pref    # Exp1（要サーバー）
    python run.py numeric    # Exp2（要サーバー）
    python run.py verbal     # Exp3（要サーバー）
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

import experiments as exp
from llm_client import HTTPLLMClient


def main():
    # 同じフォルダの .env を、環境変数より優先して読み込む
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

    name = sys.argv[1] if len(sys.argv) > 1 else "rule"

    if name == "rule":
        exp.run_rule_fixed_demo()
        exp.run_rule_hetero_demo()
    elif name == "calib":
        client = HTTPLLMClient()
        exp.calibrate_llm(client, mode="verbal")
    elif name in ("no_pref", "numeric", "verbal"):
        client = HTTPLLMClient()
        exp.run_llm_demo(client, mode=name, max_steps=10)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
