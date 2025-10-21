import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class RemoteOCRConfig:
    api_base: str
    api_key: str
    model: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: int = 30

    @staticmethod
    def from_env() -> "RemoteOCRConfig":
        return RemoteOCRConfig(
            api_base=os.getenv("REMOTE_OCR_API_BASE", ""),
            api_key=os.getenv("REMOTE_OCR_API_KEY", ""),
            model=os.getenv("REMOTE_OCR_MODEL", None),
            endpoint=os.getenv("REMOTE_OCR_ENDPOINT", None),
            timeout=int(os.getenv("REMOTE_OCR_TIMEOUT", "30")),
        )


class RemoteOCRClient:
    """
    Remote OCR Client
    
    - 统一模块注释与风格：提供通用远程 OCR（豆包/网关兼容）适配。
    - 请求体：{"image_base64": str, "model"?: str, "crop_area"?: {x1,y1,x2,y2}}
    - 返回体：优先读取 {"text": str}，兼容 {"result": str} 或 OpenAI 风格。
    """

    def __init__(self, config: RemoteOCRConfig):
        self.config = config
        if not self.config.api_base or not self.config.api_key:
            raise ValueError("远程OCR配置不完整，请设置 REMOTE_OCR_API_BASE、REMOTE_OCR_API_KEY")
        # 默认端点路径，可通过 .env 覆盖
        self.endpoint = self.config.endpoint or (self.config.api_base.rstrip('/') + '/v1/ocr')

    @staticmethod
    def from_env() -> "RemoteOCRClient":
        return RemoteOCRClient(RemoteOCRConfig.from_env())

    def _headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
        }

    def ocr(self, image_base64: str, crop_area: Optional[Dict[str, int]] = None) -> str:
        """调用远程OCR服务进行识别。
        约定请求体：
        {
          "image_base64": "...",
          "model": "可选模型名",
          "crop_area": {x1, y1, x2, y2}
        }
        约定返回：{"text": "识别结果"} 或兼容解析。
        """
        payload: Dict[str, Any] = {
            'image_base64': image_base64,
        }
        if self.config.model:
            payload['model'] = self.config.model
        if crop_area:
            payload['crop_area'] = crop_area

        resp = requests.post(self.endpoint, headers=self._headers(), data=json.dumps(payload), timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        # 常见字段解析
        if isinstance(data, dict):
            if 'text' in data:
                return data['text']
            # 某些返回可能使用 data.result 或 choices[0].message.content
            if 'result' in data and isinstance(data['result'], str):
                return data['result']
            try:
                return data.get('choices', [{}])[0].get('message', {}).get('content', '')
            except Exception:
                pass
        # 兜底：返回原始文本表示
        return json.dumps(data, ensure_ascii=False)