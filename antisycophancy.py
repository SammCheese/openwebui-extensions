"""
title: Anti-Sycophancy Anchor
author: Sammy
description: Counters positivity/confirmation bias in small instruct models (Gemma etc.) whose system prompt attention decays with context. Re-injects a condensed rule block at the END of context every turn, optionally logit-biases sycophantic opener tokens, and scrubs agreement-openers from responses.
version: 0.3.0
required_open_webui_version: 0.5.0
"""


import aiohttp
from hashlib import sha256
import json
import re
from typing import Optional
from collections import OrderedDict

from pydantic import BaseModel, Field

_ANCHOR_START = "[conduct reminder, not part of the user's message:"
_ANCHOR_END = "[/conduct reminder]"
_ANCHOR_BLOCK = re.compile(
    r"\n*\[conduct reminder, not part of the user's message:.*?" + re.escape(_ANCHOR_END) + r"\n*",
    re.DOTALL,
)
MAX_CACHE_SIZE = 16  # keep the last N system prompt digests in memory


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=20,
            description="Filter execution order (lower runs first). Keep this HIGHER than the image filters so the anchor is the last text appended to the user message.",
        )
        reminder_text: str = Field(
            default=(
                "Reminder of standing conduct rules, these override conversational "
                "momentum: State uncertainty instead of guessing facts. Disagree "
                "plainly when the user is wrong, agreement must be earned, not "
                "default. Do not open with praise or validation. Commit to a "
                "position when asked to judge. Plain prose, no lists, no em-dashes."
            ),
            description="Condensed rules injected near the end of context. Keep it under ~100 tokens, this is a nudge, not a second system prompt.",
        )
        DERIVE_FROM_SYSTEM_PROMPT: bool = Field(
            default=False,
            description="If true, the filter will attempt to derive the reminder_text from the system prompt instead of using the supplied default.",
        )
        ROUTER_BASE_URL: str = Field(
            default="http://localhost:8080/v1",
            description="Base URL of the router service used to summarize the system prompt. Leave blank to disable system prompt summarization.",
        )
        ROUTER_API_KEY: str = Field(
            default="",
            description="Optional API key for the router service.",
        )
        ROUTER_MODEL: str = Field(
            default="",
            description="Optional model name for the router service. Leave blank to use the default model.",
        )
        every_n_turns: int = Field(
            default=1,
            description="Inject every Nth user turn. 1 = every turn",
        )
        min_messages: int = Field(
            default=4,
            description="Skip injection until the conversation has at least this many messages, early context the system prompt still holds.",
        )
        as_system_message: bool = Field(
            default=False,
            description="Inject as a system message before the latest user turn instead of appending to it.",
        )
        scrub_openers: bool = Field(
            default=True,
            description="Outlet: strip sycophantic opening sentences from the model's response.",
        )
        opener_patterns: str = Field(
            default=(
                r"you're absolutely right|you are absolutely right|great question|"
                r"excellent question|fantastic question|what a great|that's a great "
                r"point|that's an excellent|i completely agree|i totally agree|"
                r"you make a great point|spot on|absolutely!|great catch|"
                r"you've hit the nail|brilliant observation|love this question"
            ),
            description="Pipe-separated regex fragments matched case-insensitively at the start of the response. Any leading sentence containing a match is dropped.",
        )
        max_scrub_sentences: int = Field(
            default=2,
            description="How many leading sentences the scrubber may remove before giving up.",
        )
        use_logit_bias: bool = Field(
            default=False,
            description="Add logit_bias to the request to suppress common praise tokens. Token IDs are tokenizer-specific, set logit_bias_json for your model.",
        )
        logit_bias_json: str = Field(
            default="{}",
            description='JSON object of token_id -> bias, e.g. {"9084": -4, "23495": -4}. Get IDs from llama-server /tokenize for strings like " Great" and " absolutely". Use -3 to -6, not -100, hard bans cause weird detours.',
        )

    def __init__(self):
        self.valves = self.Valves()
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._last_system_prompt_digest: Optional[str] = None

    # ---------- cache & status ----------

    @staticmethod
    def _digest(system_message: str) -> str:
        return sha256(system_message.encode("utf-8", "ignore")).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[str]:
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def _cache_put(self, key: str, val: str) -> None:
        self._cache[key] = val
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CACHE_SIZE:
            self._cache.popitem(last=False)

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
                        "action": "vision_prepass",
                    },
                }
            )
        except Exception:
            pass

    # ---------- inlet: system prompt summarization ----------

    async def _derive_from_system_prompt(self, emitter, system_message: str, model: str = "") -> str:
            digest = self._digest(system_message)
            cached = self._cache_get(digest)
            if cached:
                return cached

            try:
                summary = await self._summarize_system_prompt(system_message, model=model)
                if summary:
                    await self._status(emitter, f"Derived conduct reminder: {summary[:60]}...", done=True)
                    self._cache_put(digest, summary)
                    return summary
            except Exception as e:
                print(f"[anti-syco] Summarization failed: {e}")

            return self.valves.reminder_text

    async def _summarize_system_prompt(self, system_message: str, model: str = "") -> str:
            base = self.valves.ROUTER_BASE_URL.rstrip("/")
            url = f"{base}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.valves.ROUTER_API_KEY:
                headers["Authorization"] = f"Bearer {self.valves.ROUTER_API_KEY}"

            payload = {
                "model": self.valves.ROUTER_MODEL.strip() or model or "default",
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize the conduct and behavioral rules of this prompt into plain prose under 80 words. Avoid bullet points.",
                    },
                    {"role": "user", "content": system_message},
                ],
                "max_tokens": 150,
                "temperature": 0.2,
            }

            timeout = aiohttp.ClientTimeout(total=8.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:150]}")
                    data = await resp.json()

            return data["choices"][0]["message"]["content"].strip()

    # ---------- inlet: recency reinjection ----------

    async def inlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__ = None, __model__: Optional[str] = None) -> dict:
        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return body
        if len(messages) < self.valves.min_messages:
            return body

        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if self.valves.every_n_turns > 1 and user_turns % self.valves.every_n_turns:
            return body

        # Strip previous anchor injections
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and _ANCHOR_START in c:
                m["content"] = _ANCHOR_BLOCK.sub("", c).strip()
            elif isinstance(c, list):
                m["content"] = [
                    p
                    for p in c
                    if not (
                        isinstance(p, dict)
                        and p.get("type") == "text"
                        and _ANCHOR_START in (p.get("text") or "")
                    )
                ]

        body["messages"] = [
            m
            for m in messages
            if not (m.get("role") == "system" and _ANCHOR_START in str(m.get("content", "")))
        ]
        messages = body["messages"]

        reminder = self.valves.reminder_text

        # Derive reminder dynamically if requested
        if self.valves.DERIVE_FROM_SYSTEM_PROMPT and self.valves.ROUTER_BASE_URL.strip():
            model_info = __model__.get("info", {}) if isinstance(__model__, dict) else {}
            model_name = model_info.get("base_model_id") or model_info.get("name") or ""
            
            # Safe extraction of nested params
            params = model_info.get("params")
            model_sys = params.get("system", "") if isinstance(params, dict) else ""
            chat_sys = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""

            sys_parts = [p for p in (model_sys, chat_sys) if isinstance(p, str) and p.strip()]
            combined_system = "\n\n".join(sys_parts)

            if combined_system:
                sys_digest = self._digest(combined_system)
                if sys_digest != self._last_system_prompt_digest:
                    await self._status(__event_emitter__, "Deriving fresh conduct anchor...")
                    derived = await self._derive_from_system_prompt(
                        __event_emitter__, combined_system, model=model_name
                    )
                    self._last_system_prompt_digest = sys_digest
                    self._last_derived_reminder = derived
                    reminder = derived
                else:
                    reminder = self._last_derived_reminder or self.valves.reminder_text

        anchor = f"{_ANCHOR_START}]\n{reminder}\n{_ANCHOR_END}"

        if self.valves.as_system_message:
            messages.insert(-1, {"role": "system", "content": anchor})
        else:
            last = messages[-1]
            content = last.get("content")
            if isinstance(content, list):
                last["content"] = list(content) + [{"type": "text", "text": f"\n\n{anchor}"}]
            else:
                last["content"] = f"{content}\n\n{anchor}"

        # Optional token-level suppression of praise openers
        if self.valves.use_logit_bias:
            try:
                bias = json.loads(self.valves.logit_bias_json)
                if bias:
                    body["logit_bias"] = {**body.get("logit_bias", {}), **bias}
            except Exception as e:
                print(f"[anti-syco] bad logit_bias_json: {e}")

        return body

    # ---------- outlet: opener scrubbing ----------

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.scrub_openers:
            return body

        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            return body

        last = messages[-1]
        text = last.get("content")
        if not isinstance(text, str) or not text.strip():
            return body

        try:
            pattern = re.compile(self.valves.opener_patterns, re.IGNORECASE)
        except re.error as e:
            print(f"[anti-syco] Regex error: {e}")
            return body

        removed = 0
        working_text = text

        while removed < self.valves.max_scrub_sentences:
            # Handles punctuation termination or double newlines without terminal punctuation
            m = re.match(r"^\s*(.+?(?:[.!?]+|\n\n|\Z))(?:\s+|$)", working_text, re.DOTALL)
            if not m:
                break
            sentence = m.group(1).strip()
            if pattern.search(sentence):
                working_text = working_text[m.end():]
                removed += 1
            else:
                break

        if removed and working_text.strip():
            working_text = working_text.lstrip()
            last["content"] = working_text[0].upper() + working_text[1:]

        return body