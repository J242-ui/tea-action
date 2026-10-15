"""大模型客户端（OpenAI 兼容接口）。

- `QianfanClient`：百度智能云千帆 v2，用于气象灾害研判、病虫害分析与农事问答
  （伏羲/文心模型，配置 QIANFAN_* 环境变量）
- `QwenClient`：通义千问（阿里云百炼 DashScope OpenAI 兼容接口），
  用于「智能执行 Agent」的决策判断（配置 QWEN_* 环境变量）
"""
from __future__ import annotations

import requests

from . import config


BASE_URL = config.get_qianfan_base_url()
SYSTEM_PROMPT = (
    "你是茶园气象服务系统内置的农业气象专家，熟悉茶叶种植与茶园管理。"
    "请结合系统提供的实时天气、短期预报与风险研判信息，"
    "用通俗、专业的语言回答茶农关于农事操作、灾害防范、病虫害防治的问题。"
    "回答要具体可操作，可给出田间管理的建议，避免空泛。"
)
EXPLAIN_SYSTEM_PROMPT = (
    "你是茶园智能执行 Agent 的决策解释官。系统的农事作业（自动浇水、自动打药、"
    "降温喷灌、停止作业）由规则引擎决定，你的任务是用通俗、专业、简洁的中文向"
    "茶园管理者解释这次决策：为什么触发、执行了什么、预计效果与后续建议。"
    "只依据提供的数据，不得编造数字；全文控制在 100 字以内。"
)


class QianfanError(RuntimeError):
    pass


class QianfanClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = BASE_URL,
        demo: bool | None = None,
    ):
        self.api_key = api_key if api_key is not None else config.get_qianfan_api_key()
        self.model = model if model is not None else config.get_qianfan_model()
        self.base_url = base_url
        self.demo = demo if demo is not None else (not self.api_key or config.is_demo_mode())

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048,
             timeout: int = 60) -> str:
        """发送对话请求，返回模型回复文本。"""
        if self.demo:
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return self._demo_reply(last_user)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise QianfanError(f"千帆接口返回格式异常: {data}")

    def ask_agri(self, question: str, context: str) -> str:
        """农事智能问答：携带当前天气与风险上下文。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【当前茶园气象与风险上下文】\n{context}\n\n【问题】{question}"},
        ]
        return self.chat(messages)

    def assess_with_llm(self, weather_snapshot: str) -> str:
        """让大模型对给定气象快照做灾害/病虫害研判（补充规则引擎结论）。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "请基于以下气象快照，简要研判该茶园未来几天可能面临的"
                    "气象灾害与病虫害风险，并给出 2-3 条针对性建议。\n\n"
                    f"【气象快照】\n{weather_snapshot}"
                ),
            },
        ]
        return self.chat(messages, temperature=0.4, max_tokens=1024)

    def explain_decision(self, decision: dict, context: str) -> str:
        """为一次规则决策生成自然语言解释（大模型）；失败/未配置 Key 时降级为模板解释。

        decision: {decision_type, overall_level, summary, actions:[...]}
        context : 供模型参考的结构化上下文文本
        """
        if self.demo:
            return self._demo_explain(decision, context)
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": f"【决策】\n{context}\n\n请用一句话向茶园管理者解释这次决策。"},
        ]
        try:
            return self.chat(messages, temperature=0.3, max_tokens=256, timeout=8).strip()
        except Exception:  # noqa: BLE001 - 解释失败不影响决策执行
            return self._demo_explain(decision, context)

    @staticmethod
    def _demo_explain(decision: dict, context: str) -> str:
        """无 Key/调用失败时的模板解释：基于决策类型与真实摘要数据生成，演示仍可用。"""
        dtype = decision.get("decision_type", "noop")
        summary = decision.get("summary", "")
        acts = decision.get("actions") or []
        dur = next((a.get("duration_min") for a in acts if a.get("duration_min")), None)

        if dtype == "spray":
            top = ""
            if "检出" in summary:
                # 从摘要提取病害名（示例格式：检出 3 处（病害: Leaf rust of tea/…））
                s = summary.split("病害:", 1)
                if len(s) > 1:
                    top = s[1].rstrip("）)")[:24]
            head = f"【自动打药】连续多轮检出病害（{top or '茶叶病害'}），且近期无大雨、风力达标"
            tail = f"，已开启喷杆约 {dur:.0f} 分钟。" if dur else "，已开启喷杆作业。"
            return head + tail + "建议打药后 2~3 天复查叶片，关注药效与新发病灶。"
        if dtype == "irrigate":
            head = f"【自动浇水】{summary}"
            tail = f"，已开启滴灌约 {dur:.0f} 分钟。" if dur else "，已开启滴灌阀门。"
            return head + tail + "预计土壤湿度回升至 72% 后自动停灌。"
        if dtype == "irrigating":
            return f"【持续灌溉】{summary}，滴灌阀门保持开启，湿度回升至 68% 或出现大雨后自动停灌。"
        if dtype == "cool":
            return f"【降温喷灌】气温偏高且空气干燥，{summary}，已短时开启喷杆约 5 分钟降温增湿。"
        if dtype == "stop":
            reason = next((a.get("reason") for a in acts if a.get("op") in ("off", "close")), "")
            return f"【停止作业】{reason or '条件已满足'}，执行器已复位，系统继续自动巡检。"
        return f"【继续监测】本轮无需作业：{summary}，保持自动巡检，无需人工干预。"

    @staticmethod
    def _demo_reply(question: str) -> str:
        return (
            "【演示模式】当前未配置千帆 API Key，无法调用真实大模型。\n\n"
            f"您的问题是：{question}\n\n"
            "配置方式：在项目根目录创建 .env 文件（参考 .env.example），"
            "填入 QIANFAN_API_KEY 后即可获得基于伏羲/文心大模型的真实研判与问答。"
        )


class QwenClient(QianfanClient):
    """通义千问客户端（阿里云百炼 DashScope OpenAI 兼容接口）。

    用于「智能执行 Agent」的决策判断。默认配置来自 QWEN_* 环境变量：
      QWEN_API_KEY、QWEN_BASE_URL（默认 DashScope 兼容模式）、QWEN_MODEL（默认 qwen-plus）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        demo: bool | None = None,
    ):
        api_key = api_key if api_key is not None else config.get_qwen_api_key()
        model = model if model is not None else config.get_qwen_model()
        base_url = base_url if base_url is not None else config.get_qwen_base_url()
        super().__init__(api_key=api_key, model=model, base_url=base_url, demo=demo)
