"""
title: Side-Call Pre-Pass Injector
author: Sammy
version: 0.2.2
description: > Runs a lightweight "side-call" to a router model before the main model
  responds. The router's output is injected into the main model's context
  (either as a dedicated system message or invisibly appended to the user's
  message), then stripped back out after the turn completes so it doesn't
  persist or bloat future context.
required_open_webui_version: 0.5.0
"""

import re
from typing import List, Optional
import aiohttp
from pydantic import BaseModel, Field

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

def _extract_images(content, files: List[dict]) -> List[str]:
    """Collect valid image data URLs from content blocks or file metadata."""
    images = []

    # Check content blocks for base64 or absolute URLs
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                img = block.get("image_url")
                url = img.get("url") if isinstance(img, dict) else img
                if isinstance(url, str) and (url.startswith("http") or url.startswith("data:image/")):
                    images.append(url)

    # Check file attachments for direct image URLs
    if isinstance(files, list):
        for f in files:
            if isinstance(f, dict):
                url = f.get("url")
                content_type = f.get("content_type", "")
                if url and (url.startswith("http") or url.startswith("data:image/") or "image" in content_type):
                    images.append(url)

    return images

def _strip_markers(text: str) -> str:
    return _MARK_RE.sub("", text or "").strip()


SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODQgNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDcuMy4xIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjYgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTM4NCAxOTJsLTY0IDAgMCAxMjgtMTI4IDAgMCAxMjgtMTkyIDAgMC0yNS42IDE2Ni40IDAgMC0xMjggMTI4IDAgMC0xMjggODkuNiAwIDAgMjUuNnptLTI1LjYgMzguNGwwIDEyOC0xMjggMCAwIDEyOC0xNjYuNCAwIDAgMjUuNiAxOTIgMCAwLTEyOCAxMjggMCAwLTE1My42LTI1LjYgMHptMjUuNiAxOTJsLTg5LjYgMCAwIDg5LjYgMjUuNiAwIDAtNjQgNjQgMCAwLTI1LjZ6TTAgMGwwIDM4NCAxMjggMCAwLTEyOCAxMjggMCAwLTEyOCAxMjggMCAwLTEyOC0zODQgMHoiLz48L3N2Zz4="


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Filter execution order. Lower values run first."
        )
        ROUTER_BASE_URL: str = Field(
            default="http://localhost:8080/v1",
            description="Base URL of the llama.cpp OpenAI-compatible endpoint for the side-call/router model, e.g. http://localhost:8081/v1 (no trailing slash needed).",
        )
        ROUTER_API_KEY: Optional[str] = Field(
            default=None,
            description="API key for the router endpoint, sent as 'Authorization: Bearer <key>'. Leave blank if llama-server has no key configured.",
        )
        ROUTER_MODEL: Optional[str] = Field(
            default=None,
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
            default=False,
            description="Default injection mode. True: inject as a standalone system message placed right before the user's turn. False: append invisibly to the end of the user's message. Users can override this per-chat via the filter chip's valve modal.",
        )
        ENABLE_THINKING: str = Field(
            default="auto",
            json_schema_extra={"enum": ["auto", "off", "on"]},
            description=(
                "Sets chat_template_kwargs.enable_thinking for thinking-capable "
                "models (Qwen3-VL etc.), so the whole max_tokens budget goes to "
                "the actual enumeration. 'auto' = don't send the field at all. "
                "Needs llama-server running with --jinja; harmless no-op for "
                "models whose template ignores it."
            ),
        )
        USE_FILE_CONTEXT: bool = Field(
            default=True,
            description="If true, the side-call will receive the user's file context (if any). If false, the side-call will only see the user's text message.",
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
        ENABLED: bool = Field(
            default=True,
            description="If false, the prepass is skipped entirely.",
        )

    class UserValves(BaseModel):
        INJECT_AS_SYSTEM_MESSAGE: bool = Field(
            default=False,
            description="Override the admin default: inject as a system message (on) or append invisibly to your message (off).",
        )
        ENABLED: bool = Field(
            default=True,
            description="Override the admin default: If false, the prepass is skipped entirely.",
        )
        USE_FILE_CONTEXT: bool = Field(
            default=True,
            description="Override the admin default: If false, the side-call will only see your text message, not any file context.",
        )
        SIDE_CALL_SYSTEM_PROMPT: Optional[str] = Field(
            default=None,
            description="Override the admin default system prompt for the side-call. Leave blank to use the admin default.",
        )
        SIDE_CALL_HISTORY_TURNS: Optional[int] = Field(
            default=None,
            description="Override the admin default number of prior user/assistant turn-pairs to send to the side-call as context. Set to 0 to use the admin default.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = SVG

    # ---------- user valves ----------

    def user_valves(self, __user__: dict) -> dict:
        uv = __user__.get("valves") if isinstance(__user__, dict) else None
        
        if uv is not None:
            if hasattr(uv, "model_dump"):
                uv_dict = uv.model_dump()
            elif hasattr(uv, "dict"):
                uv_dict = uv.dict()
            elif isinstance(uv, dict):
                uv_dict = uv
            else:
                uv_dict = {}

            return uv_dict
        
        return {}

    # --------- status ----------
    @staticmethod
    async def _status(emitter, description: str, done: bool = False) -> None:
        if not emitter:
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                        "action": "side-call pre-pass",
                    },
                }
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- router call

    def _clean_model_id(self, model_id: Optional[str]) -> Optional[str]:
        """Clean up the model ID to avoid sending empty strings or whitespace."""
        if model_id is None:
            return None
        cleaned = model_id.strip().strip('"').strip("'")
        return cleaned if cleaned else None

    async def _call_router(self, side_messages: List[dict], model: Optional[str]) -> str:
        base = self.valves.ROUTER_BASE_URL.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.valves.ROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.ROUTER_API_KEY}"

        payload = {
            "model": model or "gpt-4o-mini",
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
        if not isinstance(body, dict):
            return body

        user_valves = self.user_valves(__user__)

        # Safe resolution: fall back to admin valve if user valve is None
        enabled = user_valves.get("ENABLED") if user_valves.get("ENABLED") is not None else self.valves.ENABLED
        if not enabled:
            return body

        inject_as_system = (
            user_valves.get("INJECT_AS_SYSTEM_MESSAGE")
            if user_valves.get("INJECT_AS_SYSTEM_MESSAGE") is not None
            else self.valves.INJECT_AS_SYSTEM_MESSAGE
        )
        use_files = (
            user_valves.get("USE_FILE_CONTEXT")
            if user_valves.get("USE_FILE_CONTEXT") is not None
            else self.valves.USE_FILE_CONTEXT
        )
        user_prompt = (user_valves.get("SIDE_CALL_SYSTEM_PROMPT") or "").strip()
        system_prompt = user_prompt if user_prompt else self.valves.SIDE_CALL_SYSTEM_PROMPT

        user_turns = user_valves.get("SIDE_CALL_HISTORY_TURNS")
        if user_turns is None:
            user_turns = self.valves.SIDE_CALL_HISTORY_TURNS

        messages = body.get("messages")
        if not messages:
            return body

        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return body
        
        raw_content = last_msg.get("content")
        user_text = _strip_markers(_extract_text(raw_content))
        images = _extract_images(raw_content, last_msg.get("files", [])) if use_files else []

        if not user_text and not images:
            return body

        if self.valves.EMIT_STATUS and __event_emitter__:
            await self._status(__event_emitter__, "Running side-call pre-pass...")

        side_messages = [{"role": "system", "content": system_prompt}]

        if user_turns > 0:
            pair_count = user_turns * 2
            trimmed = []
            for m in messages[:-1]:
                role = m.get("role")
                if role not in ("user", "assistant"):
                    continue
                text = _strip_markers(_extract_text(m.get("content")))
                if text:
                    trimmed.append({"role": role, "content": text})
            side_messages.extend(trimmed[-pair_count:])

        if images:
            side_user_content = [{"type": "text", "text": user_text}]
            for img_url in images[:3]:
                side_user_content.append({"type": "image_url", "image_url": {"url": img_url}})
            side_messages.append({"role": "user", "content": side_user_content})
        else:
            side_messages.append({"role": "user", "content": user_text})

        model_name = (
            self._clean_model_id(self.valves.ROUTER_MODEL) 
            or self._clean_model_id(body.get("model"))
            or None
        )

        try:
            result = await self._call_router(side_messages, model_name)
        except Exception as e:
            if self.valves.EMIT_STATUS and __event_emitter__:
                await self._status(__event_emitter__, f"Pre-pass failed ({e}); continuing without it", done=True)
            if self.valves.ON_ERROR == "abort":
                raise
            return body

        result = (result or "").strip()

        if self.valves.EMIT_STATUS and __event_emitter__:
            await self._status(__event_emitter__, "Pre-pass complete", done=True)

        if not result:
            return body

        injected_block = (
            f"{MARK_START}\n{self.valves.MAIN_PROMPT_PREFIX}\n{result}\n{MARK_END}"
        )

        if inject_as_system:
            messages.insert(
                len(messages) - 1, {"role": "system", "content": injected_block}
            )
        else:
            if isinstance(raw_content, list):
                raw_content.append({"type": "text", "text": f"\n\n{injected_block}"})
            else:
                last_msg["content"] = f"{user_text}\n\n{injected_block}"

        if __metadata__ is not None:
            __metadata__["_sidecall_injected"] = True

        return body
