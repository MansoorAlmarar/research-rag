"""Optional model call through OpenRouter. The key comes from the OPENROUTER_API_KEY environment variable and is never printed."""
import os
import time

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"


def make_llm(model, timeout=60, retries=1):
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    def call(system, user):
        last = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(URL, timeout=timeout, headers={"Authorization": f"Bearer {key}"},
                                  json={"model": model, "temperature": 0, "max_tokens": 600,
                                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                last = RuntimeError(f"HTTP {r.status_code}")
            except requests.RequestException as e:
                last = e
            time.sleep(2 * (attempt + 1))
        raise last

    return call
