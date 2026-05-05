import json
import logging
import re
import time

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Config):
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model = config.model
        self.temperature = config.temperature
        self.is_gemma = config.is_gemma
        self.max_retries = 3
        self.base_delay = 2.0

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list:
        if self.is_gemma:
            return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=self._build_messages(system_prompt, user_prompt),
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict | list:
        for attempt in range(self.max_retries):
            raw = self.chat(system_prompt, user_prompt)
            try:
                return self._parse_json(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to parse JSON after {self.max_retries} attempts. Raw: {raw[:500]}")
                    raise

    def chat_structured(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        """Call the LLM with a JSON schema constraint (guided decoding).

        Works with vLLM and Ollama 0.5+. The model is forced to produce
        valid JSON matching the provided schema, eliminating parse errors.
        Retries still protect against network / transient failures.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": schema,
            },
        }

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format=response_format,
                    messages=self._build_messages(system_prompt, user_prompt),
                    max_tokens=1024,
                )
                if response.choices[0].finish_reason == "length":
                    raise ValueError(
                        "output truncated (finish_reason=length) — input too long or max_tokens too small"
                    )
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
            except Exception as e:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                else:
                    raise

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return json.loads(cleaned)
