"""研究室LLMサーバーへの接続テスト。

実行: python test_connection.py
段階的に切り分けられるよう、(1)疎通・モデル確認 → (2)実推論 の順に検証する。
"""

import sys
from dotenv import load_dotenv
from llm_client import LabLLMClient


def main() -> None:
    load_dotenv()  # 同じディレクトリの .env を読み込む

    # 環境変数の存在チェック
    try:
        client = LabLLMClient()
    except KeyError as e:
        print(f"[NG] 環境変数が見つかりません: {e}")
        print("     .env に LLM_BASE_URL / LLM_API_KEY / LLM_MODEL を設定してください。")
        sys.exit(1)

    print(f"接続先 : {client.base_url}")
    print(f"モデル : {client.model}")
    print("-" * 48)

    # (1) 疎通 & モデル存在確認
    try:
        models = client.list_models()
        print(f"[OK] サーバーに接続できました。公開モデル: {models}")
        if client.model not in models:
            print(f"[警告] 指定モデル '{client.model}' が一覧にありません。モデル名を確認してください。")
    except Exception as e:
        print(f"[NG] サーバーに接続できません: {type(e).__name__}: {e}")
        print("     URL / API Key / ネットワーク(VPN等) を確認してください。")
        print("     ※ /v1/models が404の場合、OpenAI互換ではない可能性があります。")
        sys.exit(1)

    # (2) 実際に推論できるか
    try:
        answer = client.chat(
            system_prompt="あなたは簡潔に答えるアシスタントです。",
            user_prompt="「接続テスト成功」とだけ答えてください。",
        )
        print(f"[OK] 推論に成功しました。応答: {answer!r}")
    except Exception as e:
        print(f"[NG] 推論に失敗しました: {type(e).__name__}: {e}")
        sys.exit(1)

    print("-" * 48)
    print("すべてのテストに合格しました。")


if __name__ == "__main__":
    main()
