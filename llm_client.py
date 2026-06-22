"""LLMクライアント群。

ノートブックの LLMClient(ABC) インターフェース（generate(prompt) -> str）を
維持しつつ、研究室サーバー(oMLX / OpenAI互換)に接続する HTTPLLMClient を主役にする。

- HTTPLLMClient : 本実験用。oMLX へHTTPで推論。thinking は既定でオン。
                  decide() 側が最後に現れた stay/move を結論として拾うため、
                  思考過程が本文に出ても判定できる。
- DummyLLMClient: LLMなしで判定器やシミュレーションの流れを確認する用。
- LocalLLMClient: Colab等でローカルGPU推論する場合の代替（transformersを遅延import）。

接続テスト互換のため list_models() / chat() も残し、LabLLMClient を別名にしている。
"""

import os
from abc import ABC, abstractmethod
from openai import OpenAI


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """1つのプロンプト文字列を受け取り、応答テキストを返す。"""
        ...


class DummyLLMClient(LLMClient):
    """LLMなしでも判定器のテストができるダミー。"""

    def __init__(self, default: str = "stay"):
        self.default = default

    def generate(self, prompt: str) -> str:
        return self.default


class HTTPLLMClient(LLMClient):
    """研究室サーバー(oMLX / OpenAI互換)へHTTPで接続するクライアント。

    temperature は再現性を厳密に固定する必要がないため既定 0.7。
    複数回実行して傾向として再現性を見る運用に合わせている。
    thinking は既定でオン（enable_thinking=True）。明示的に False にしたときだけ
    Qwen3 の "/no_think" を付与する（oMLX 側設定で thinking 制御している場合は無効）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,  # thinking が結論まで到達できるよう十分に確保
        enable_thinking: bool = True,
    ):
        self.base_url = base_url or os.environ["LLM_BASE_URL"]
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self.model = model or os.environ["LLM_MODEL"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # --- 接続テスト用に残す ---
    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content

    # --- シミュレーションで実際に使う入口 ---
    def generate(self, prompt: str) -> str:
        if not self.enable_thinking:
            prompt = prompt + " /no_think"  # Qwen3のthinking抑制スイッチ
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content


# 接続テスト(test_connection.py)との後方互換
LabLLMClient = HTTPLLMClient


class LocalLLMClient(LLMClient):
    """Colab等でローカルGPU推論する場合の代替（研究室サーバー利用時は不要）。"""

    def __init__(self, model_name: str = "Qwen/Qwen3-4B",
                 max_new_tokens: int = 1024, enable_thinking: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        print("Loaded.")

    def generate(self, prompt: str) -> str:
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
