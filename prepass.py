"""
title: Side-Call Pre-Pass Injector
author: Sammy
version: 0.1.0
description: > Runs a lightweight "side-call" to a router model before the main model
  responds. The router's output is injected into the main model's context
  (either as a dedicated system message or invisibly appended to the user's
  message), then stripped back out after the turn completes so it doesn't
  persist or bloat future context.
required_open_webui_version: 0.5.0
"""

import re
import aiohttp
from pydantic import BaseModel, Field
from typing import Optional, List

# Zero-width-space + HTML-comment markers: invisible in rendered markdown/HTML,
# but a reliable, greppable span for outlet() to find and strip.
MARK_START = "\u200b<!--owf:sidecall:start-->"
MARK_END = "<!--owf:sidecall:end-->\u200b"
_MARK_RE = re.compile(re.escape(MARK_START) + r".*?" + re.escape(MARK_END), re.DOTALL)


def _extract_text(content) -> str:
    """Pull plain text out of either a string or an OpenAI-style content-block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _strip_markers(text: str) -> str:
    return _MARK_RE.sub("", text or "").strip()


class Filter:
    class Valves(BaseModel):
        ROUTER_BASE_URL: str = Field(
            default="http://localhost:8080/v1",
            description="Base URL of the llama.cpp OpenAI-compatible endpoint for the side-call/router model, e.g. http://localhost:8081/v1 (no trailing slash needed).",
        )
        ROUTER_API_KEY: str = Field(
            default="",
            description="API key for the router endpoint, sent as 'Authorization: Bearer <key>'. Leave blank if llama-server has no key configured.",
        )
        ROUTER_MODEL: str = Field(
            default="",
            description="Model name sent in the side-call's 'model' field. Optional — most single-model llama-server setups ignore this; leave blank to send a harmless placeholder.",
        )
        SIDE_CALL_SYSTEM_PROMPT: str = Field(
            default="You are a preprocessing assistant. Read the user's message and produce brief, useful notes for the main assistant that will respond next.",
            description="System prompt for the side-call model. This is where you define the pre-pass task (classify intent, extract entities, rewrite the query, pull reminders, etc.).",
        )
        MAIN_PROMPT_PREFIX: str = Field(
            default="The following context was provided by a pre-pass assistant. Use it to inform your response, but do not repeat it verbatim unless explicitly asked.",
            description="Text prefixed to the side-call's output before it's injected into the main model's context.",
        )
        INJECT_AS_SYSTEM_MESSAGE: bool = Field(
            default=True,
            description="Default injection mode. True: inject as a standalone system message placed right before the user's turn. False: append invisibly to the end of the user's message. Users can override this per-chat via the filter chip's valve modal.",
        )
        ENABLE_THINKING: str = Field(
            default="off",
            json_schema_extra={"enum": ["auto", "off", "on"]},
            description=(
                "Sets chat_template_kwargs.enable_thinking for thinking-capable "
                "models (Qwen3-VL etc.), so the whole max_tokens budget goes to "
                "the actual enumeration. 'auto' = don't send the field at all. "
                "Needs llama-server running with --jinja; harmless no-op for "
                "models whose template ignores it."
            ),
        )
        SIDE_CALL_HISTORY_TURNS: int = Field(
            default=0,
            ge=0,
            description="Number of prior user/assistant turn-pairs to send to the side-call as context. 0 = only the latest user message is sent.",
        )
        SIDE_CALL_MAX_TOKENS: int = Field(
            default=512, description="max_tokens for the side-call request."
        )
        SIDE_CALL_TEMPERATURE: float = Field(
            default=0.3, description="temperature for the side-call request."
        )
        REQUEST_TIMEOUT: float = Field(
            default=30.0,
            description="Timeout in seconds for the side-call HTTP request.",
        )
        ON_ERROR: str = Field(
            default="skip",
            description="'skip': if the side-call fails, continue the main turn without injected context. 'abort': cancel the whole turn.",
            json_schema_extra={"enum": ["skip", "abort"]},
        )
        EMIT_STATUS: bool = Field(
            default=True,
            description="Show status messages in the chat UI while the side-call runs.",
        )
        priority: int = Field(
            default=0, description="Filter execution order. Lower values run first."
        )

    class UserValves(BaseModel):
        INJECT_AS_SYSTEM_MESSAGE: Optional[bool] = Field(
            default=None,
            description="Override the admin default: inject as a system message (on) or append invisibly to your message (off). Leave unset to use the admin default.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = (
            True  # user-controllable chip; clicking it opens the UserValves modal above
        )

    # ---------------------------------------------------------------- router call

    async def _call_router(self, side_messages: List[dict], model: str) -> str:
        base = self.valves.ROUTER_BASE_URL.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.valves.ROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.ROUTER_API_KEY}"

        payload = {
            "model": self.valves.ROUTER_MODEL.strip() or model or "default",
            "messages": side_messages,
            "max_tokens": self.valves.SIDE_CALL_MAX_TOKENS,
            "temperature": self.valves.SIDE_CALL_TEMPERATURE,
            "stream": False,
        }

        if self.valves.ENABLE_THINKING != "auto":
          payload["chat_template_kwargs"] = {
            "enable_thinking": self.valves.ENABLE_THINKING == "on"
          }

        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    raise RuntimeError(f"router HTTP {resp.status}: {body_text[:200]}")
                data = await resp.json()

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected router response shape: {e}")

    # ---------------------------------------------------------------- inlet

    async def inlet(
        self,
        body: dict,
        __event_emitter__=None,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages")
        if not messages:
            return body

        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return body

        user_text = _strip_markers(_extract_text(last_msg.get("content")))
        if not user_text:
            return body

        if self.valves.EMIT_STATUS and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Running side-call pre-pass...",
                        "done": False,
                    },
                }
            )

        side_messages = [
            {"role": "system", "content": self.valves.SIDE_CALL_SYSTEM_PROMPT}
        ]

        if self.valves.SIDE_CALL_HISTORY_TURNS > 0:
            pair_count = self.valves.SIDE_CALL_HISTORY_TURNS * 2
            trimmed = []
            for m in messages[:-1]:
                role = m.get("role")
                if role not in ("user", "assistant"):
                    continue
                text = _strip_markers(_extract_text(m.get("content")))
                if text:
                    trimmed.append({"role": role, "content": text})
            side_messages.extend(trimmed[-pair_count:])

        side_messages.append({"role": "user", "content": user_text})

        try:
            if __model__ and isinstance(__model__, dict):
                model_name = __model__["info"].get("base_model_id")

            result = await self._call_router(side_messages, model_name)
        except Exception as e:
            if self.valves.EMIT_STATUS and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Pre-pass failed ({e}); continuing without it",
                            "done": True,
                        },
                    }
                )
            if self.valves.ON_ERROR == "abort":
                raise
            return body

        result = (result or "").strip()

        if self.valves.EMIT_STATUS and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Pre-pass complete", "done": True},
                }
            )

        if not result:
            return body

        inject_as_system = self.valves.INJECT_AS_SYSTEM_MESSAGE
        user_valves = (__user__ or {}).get("valves")
        if (
            isinstance(user_valves, self.UserValves)
            and user_valves.INJECT_AS_SYSTEM_MESSAGE is not None
        ):
            inject_as_system = user_valves.INJECT_AS_SYSTEM_MESSAGE

        injected_block = (
            f"{MARK_START}\n{self.valves.MAIN_PROMPT_PREFIX}\n{result}\n{MARK_END}"
        )

        if inject_as_system:
            # Terminal context position: right before the user's turn, not the very top.
            messages.insert(
                len(messages) - 1, {"role": "system", "content": injected_block}
            )
        else:
            content = last_msg.get("content")
            if isinstance(content, list):
                content.append({"type": "text", "text": f"\n\n{injected_block}"})
            else:
                last_msg["content"] = f"{_extract_text(content)}\n\n{injected_block}"

        if __metadata__ is not None:
            __metadata__["_sidecall_injected"] = True

        return body

    # ---------------------------------------------------------------- outlet

    async def outlet(self, body: dict, __metadata__: Optional[dict] = None) -> dict:
        messages = body.get("messages")
        if not messages:
            return body

        cleaned = []
        for m in messages:
            content = m.get("content")

            if isinstance(content, str):
                if MARK_START in content:
                    new_text = _strip_markers(content)
                    if not new_text:
                        continue  # whole message was our injected system block: drop it
                    m["content"] = new_text

            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and MARK_START in block.get("text", "")
                    ):
                        stripped = _strip_markers(block["text"])
                        if stripped:
                            block["text"] = stripped
                            new_blocks.append(block)
                        # else: drop this block, it was pure injection
                    else:
                        new_blocks.append(block)
                m["content"] = new_blocks

            cleaned.append(m)

        body["messages"] = cleaned
        return body
