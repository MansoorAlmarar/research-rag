"""Model calls. Two providers behind one function `llm(system, user) -> str`.

    make_llm("nvidia/nemotron-3-ultra-550b-a55b:free")   OpenRouter. Needs OPENROUTER_API_KEY (never printed). Free models work with $0 credit.
    make_llm("claude-cli:haiku")                          the local `claude` CLI on your Claude subscription. No API key, no per-call dollars,
                                                          but it uses plan usage. `claude-cli:sonnet` and `claude-cli:opus` also work.

The CLI is run with all tools off, in an empty temporary folder (so no project instructions are loaded), with the
prompt on stdin. Set CLAUDE_EXE to the full path of claude.exe if it is not found automatically.
"""
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
CLI_PREFIX = "claude-cli:"
_WINDOWS_PATTERNS = [r"%LOCALAPPDATA%\Claude-Data\claude_*\claude-code\*\*\claude.exe",
                     r"%APPDATA%\Claude\claude-code\*\*\claude.exe",
                     r"%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\claude-code\*\*\claude.exe"]


def find_claude_exe():
    """CLAUDE_EXE, then `claude` on PATH, then the desktop app's bundled copy (newest version)."""
    env = os.environ.get("CLAUDE_EXE", "").strip()
    if env and os.path.exists(env):
        return env
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    found = [p for pat in _WINDOWS_PATTERNS for p in glob.glob(os.path.expandvars(pat))]

    def version(p):
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        return [int(x) if x.isdigit() else 0 for x in name.split(".")]

    return sorted(found, key=version)[-1] if found else None


def cli_command(exe, model, system):
    return [exe, "-p", "--model", model, "--effort", "low", "--output-format", "json", "--system-prompt", system,
            "--tools", "", "--disable-slash-commands", "--no-session-persistence"]


def make_cli_llm(model, timeout=180):
    exe = find_claude_exe()
    if not exe:
        raise RuntimeError("claude CLI not found (set CLAUDE_EXE)")

    def call(system, user):
        with tempfile.TemporaryDirectory() as empty:
            r = subprocess.run(cli_command(exe, model, system), input=user, capture_output=True, text=True,
                               encoding="utf-8", cwd=empty, timeout=timeout)
        try:
            data = json.loads(r.stdout)
        except ValueError:
            raise RuntimeError(f"claude CLI exit {r.returncode}: {r.stderr.strip()[-160:] or 'no JSON output'}") from None
        if data.get("is_error") or r.returncode != 0:
            raise RuntimeError(f"claude CLI error: {str(data.get('result', ''))[:160]}")
        return data.get("result", "")

    return call


def make_llm(model, timeout=60, retries=1):
    if model.startswith(CLI_PREFIX):
        return make_cli_llm(model[len(CLI_PREFIX):] or "haiku")
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
                # Only a rate limit or a server error can succeed on a second try. A 400, 401 or 404 will not.
                if r.status_code != 429 and r.status_code < 500:
                    break
            except requests.RequestException as e:
                last = e
            if attempt < retries:  # no wait after the last try
                time.sleep(2 * (attempt + 1))
        raise last

    return call
