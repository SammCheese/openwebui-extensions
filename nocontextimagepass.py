"""
title: Context-Free Vision Pre-Pass
author: Sammy
description: Sends images to llama-server in an isolated context for unbiased enumeration, then injects the result as text so the main conversation can't overwrite what the model saw.
version: 0.6.0
required_open_webui_version: 0.5.0
"""

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from typing import Optional

import aiohttp
from pydantic import BaseModel, Field

_INJECT_MARKER = "[Independent visual inventory of image"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=10,
            description="Filter execution order (lower runs first). Keep this between the terminal mirror (0) and the anti-sycophancy anchor (20).",
        )
        llama_url: str = Field(
            default="http://127.0.0.1:5001/v1/chat/completions",
            description="llama-server chat completions endpoint",
        )
        llama_model: str = Field(
            default="",
            description="model to use in router mode, leave empty for the currently loaded model",
        )
        api_key: str = Field(
            default="",
            description="API key for llama-server (--api-key value)",
        )
        id_slot: int = Field(
            default=-1,
            description="Slot for the vision pass. -1 = let server pick. If you run --parallel 2, set this to 1 so your main chat's cache in slot 0 survives.",
        )
        enumeration_prompt: str = Field(
            default=(
                "Enumerate everything physically visible in this image: objects, "
                "people, physiques, anatomical details, colors, shapes, counts, "
                "spatial layout, and any readable text. Be exhaustive and neutral. "
                "Do not interpret, do not speculate about purpose or story, do not "
                "omit anything as irrelevant. Plain prose."
            ),
            description="Neutral prompt sent with the isolated image",
        )
        max_tokens: int = Field(
            default=768, description="Max tokens for the enumeration"
        )
        thinking: str = Field(
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
        regenerate: bool = Field(
            default=False,
            description="If true, the vision pass is re-run on every retry/regeneration. If false, the pass is only run once per image digest and cached for subsequent retries.",
        )
        temperature: float = Field(
            default=0.2, description="Low temp keeps enumeration factual"
        )
        keep_image: bool = Field(
            default=True,
            description="Keep the original image in the message alongside the injected description. Off = replace image with description only (fully context-proof, but the main pass can no longer look at pixels at all). If off, any filter that needs the image (e.g. the terminal mirror) must run BEFORE this one.",
        )
        concurrent: bool = Field(
            default=False,
            description="Describe multiple images in parallel. Only enable if llama-server runs with --parallel > 1 (and mind id_slot: a fixed slot serializes anyway).",
        )
        timeout: int = Field(
            default=180, description="Seconds to wait for the vision pass"
        )
        cache_size: int = Field(
            default=128,
            description="How many digest→description entries to keep, so a retried or edited message doesn't re-run the vision pass on the same image",
        )

    def __init__(self):
        self.valves = self.Valves()
        # image digest -> description. Retrying/regenerating a turn resends the
        # same image; the digest matches and we skip the API call entirely.
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        # Last failure reason, surfaced in the UI status so users don't have
        # to tail the server log to find out why the pass produced nothing.
        self._last_error: Optional[str] = None

    # ---------- cache ----------

    @staticmethod
    def _digest(image_part: dict) -> str:
        url = (image_part.get("image_url") or {}).get("url", "")
        return hashlib.sha256(url.encode("utf-8", "ignore")).hexdigest()[:16]

    def _cache_get(self, key: str) -> Optional[str]:
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def _cache_put(self, key: str, val: str) -> None:
        self._cache[key] = val
        self._cache.move_to_end(key)
        while len(self._cache) > self.valves.cache_size:
            self._cache.popitem(last=False)

    # ---------- vision pass ----------

    def _extract_images(self, content) -> list:
        """Return list of image_url dicts from an Open WebUI message content."""
        if not isinstance(content, list):
            return []
        return [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]

    async def _describe(
        self, session: aiohttp.ClientSession, image_part: dict, model: Optional[str] = None
    ) -> Optional[str]:
        """One isolated API call: image + neutral prompt, zero history."""
        key = self._digest(image_part)
        cached = self._cache_get(key)
        if cached is not None and not self.valves.regenerate:
            return cached

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_part,
                        {"type": "text", "text": self.valves.enumeration_prompt},
                    ],
                }
            ],
            "max_tokens": self.valves.max_tokens,
            "temperature": self.valves.temperature,
        }
        if self.valves.thinking != "auto":
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.valves.thinking == "on"
            }
        if self.valves.id_slot >= 0:
            payload["id_slot"] = self.valves.id_slot

        if self.valves.llama_model:
            payload["model"] = self.valves.llama_model
        elif model:
            payload["model"] = model

        try:
            async with session.post(
                self.valves.llama_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.valves.api_key}",
                    "Content-Type": "application/json",
                },
            ) as r:
                raw = await r.text()
                if r.status != 200:
                    self._fail(f"HTTP {r.status}: {raw[:300]}")
                    return None
            data = json.loads(raw)
        except asyncio.TimeoutError:
            self._fail(
                f"timed out after {self.valves.timeout}s — raise the timeout "
                f"valve if generation legitimately takes longer"
            )
            return None
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            return None

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        # Some servers return content as a list of parts instead of a string
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        desc = (content or "").strip()
        # Strip chain-of-thought. A leading <think> with no closing tag means
        # the model spent its entire budget thinking; there is no answer.
        desc = _THINK_RE.sub("", desc).strip()
        if desc.startswith("<think>"):
            desc = ""

        if not desc:
            finish = choice.get("finish_reason")
            reasoning = (msg.get("reasoning_content") or "").strip()
            hint = ""
            if reasoning or finish == "length":
                hint = (
                    " — model used its whole max_tokens budget on thinking; "
                    "set the thinking valve to 'off' (needs --jinja) or raise "
                    "max_tokens"
                )
            self._fail(f"empty answer (finish_reason={finish}){hint}")
            return None

        if not self.valves.regenerate:
            self._cache_put(key, desc)
        return desc

    def _fail(self, reason: str) -> None:
        self._last_error = reason
        print(f"[vision-prepass] failed: {reason}")

    # ---------- filter entrypoint ----------

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

    async def inlet(
        self,
        body: dict,
        __event_emitter__=None,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages", [])
        if not messages:
            return body

        # Only process the newest user message; older images were handled on their turn.
        last = messages[-1]
        if last.get("role") != "user":
            return body

        content = last.get("content")
        images = self._extract_images(content)
        if not images:
            return body

        # Already injected (e.g. an edited message that kept the injected text):
        # skip so we don't stack duplicate inventories.
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and _INJECT_MARKER in (part.get("text") or "")
            ):
                return body
            
        if __model__ and "info" in __model__:
            model = __model__["info"].get("base_model_id") or None
        else:
            model = None

        self._last_error = None

        # On a retry/regeneration every digest hits the cache, so don't show an
        # "analyzing" status for work that won't happen.
        misses = sum(1 for img in images if self._cache_get(self._digest(img)) is None and not self.valves.regenerate)

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.valves.concurrent and len(images) > 1:
                if misses:
                    await self._status(
                        __event_emitter__,
                        f"Running context-free vision pass on {misses} image(s)…",
                    )
                raw = await asyncio.gather(
                    *(self._describe(session, img, model=model) for img in images),
                    return_exceptions=True,
                )
                for i, r in enumerate(raw, 1):
                    if isinstance(r, BaseException):
                        print(f"[vision-prepass] image {i} raised: {r!r}")
                raw = [r if isinstance(r, str) else None for r in raw]
            else:
                raw = []
                for i, img in enumerate(images, 1):
                    if self._cache_get(self._digest(img)) is None and not self.valves.regenerate:
                        await self._status(
                            __event_emitter__,
                            f"Context-free vision pass: image {i}/{len(images)}…",
                        )
                    raw.append(await self._describe(session, img, model=model))

        descriptions = [
            f"{_INJECT_MARKER} {i}, produced with no "
            f"conversation context. Treat this as ground truth for what the "
            f"image contains:]\n{desc}"
            for i, desc in enumerate(raw, 1)
            if desc
        ]
        if not descriptions:
            reason = f": {self._last_error}" if self._last_error else ""
            await self._status(
                __event_emitter__,
                f"Vision pre-pass failed{reason} — images passed through untouched",
                done=True,
            )
            return body

        cached_note = "" if misses else " (cached)"
        await self._status(
            __event_emitter__,
            f"Visual inventory injected for {len(descriptions)}/{len(images)} "
            f"image(s){cached_note}",
            done=True,
        )

        injected = "\n\n".join(descriptions)
        new_content = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and not self.valves.keep_image
            ):
                continue
            new_content.append(part)
        new_content.append({"type": "text", "text": injected})
        last["content"] = new_content

        return body
