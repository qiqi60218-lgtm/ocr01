"""
LLM Client

- 统一模块注释与风格：用于文本纠错与结构化抽取（OpenAI 兼容风格）。
- 默认端点：`/v1/chat/completions`，可通过 `.env` 设置自定义端点。
- 鉴权：`Authorization: Bearer <LLM_API_KEY>`。
"""
import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    provider: str
    api_base: str
    api_key: str
    model: str
    endpoint: Optional[str] = None  # 可选自定义端点（优先级高于默认）
    temperature: float = 0.2
    timeout: int = 30

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "doubao"),
            api_base=os.getenv("LLM_API_BASE", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", ""),
            endpoint=os.getenv("LLM_ENDPOINT", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        )


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if not self.config.api_base or not self.config.api_key or not self.config.model:
            raise ValueError("LLM配置不完整，请设置 LLM_API_BASE、LLM_API_KEY、LLM_MODEL")

        # 统一默认端点（OpenAI兼容风格）
        self.endpoint = self.config.endpoint or \
            (self.config.api_base.rstrip('/') + '/v1/chat/completions')

    @staticmethod
    def from_env() -> "LLMClient":
        return LLMClient(LLMConfig.from_env())

    def _headers(self) -> Dict[str, str]:
        # 大多数提供商都使用 Bearer 令牌；如有不同可在网关层适配
        return {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            self.endpoint,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def fix_text(self, text: str, extra_instructions: Optional[str] = None) -> str:
        """使用LLM进行OCR文本纠错，保持原意与格式，适配中英文混排。"""
        system_prompt = (
            "你是一个OCR文本修复助手。\n"
            "任务：在尽量保留原文意思与段落结构的前提下，修复常见的OCR错误（错别字、空格、标点、大小写、连字、乱码）。\n"
            "规范：\n"
            "- 不要臆造或添加缺失内容。\n"
            "- 保持原有换行与段落结构。\n"
            "- 对中英文混排做合理的空格与标点修正。\n"
            "- 输出仅为修复后的纯文本。"
        )
        if extra_instructions:
            system_prompt += ("\n附加说明：" + extra_instructions)

        payload = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text},
            ]
        }
        data = self._post(payload)
        # OpenAI风格解析
        try:
            return data['choices'][0]['message']['content']
        except Exception:
            # 尝试兼容其他风格
            return json.dumps(data, ensure_ascii=False)

    def extract_structured(self, text: str, schema_hint: Optional[Dict[str, Any]] = None,
                           instruction: Optional[str] = None) -> Dict[str, Any]:
        """使用LLM进行结构化抽取，输出严格的JSON对象。"""
        system_prompt = (
            "你是一个结构化信息抽取助手。\n"
            "请从用户提供的OCR文本中抽取关键信息并以JSON格式返回。\n"
            "要求：\n"
            "- 严格输出JSON对象，不要输出多余文字。\n"
            "- 若字段缺失，请填为null或空字符串。\n"
            "- 保留原文数值与日期格式。"
        )
        if instruction:
            system_prompt += ("\n任务说明：" + instruction)
        if schema_hint:
            system_prompt += ("\n示例Schema：" + json.dumps(schema_hint, ensure_ascii=False))

        payload = {
            'model': self.config.model,
            'temperature': 0.1,  # 抽取任务尽量降低随机性
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text},
            ]
        }
        data = self._post(payload)
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        try:
            return json.loads(content)
        except Exception:
            # 尝试去掉可能的markdown包裹
            content = content.strip()
            if content.startswith('```'):
                content = content.strip('`')
                # 可能包含json语言标记
                content = content.replace('json', '')
            try:
                return json.loads(content)
            except Exception:
                # 返回原始文本，交由前端降级处理
                return {"raw": content}
