"""
title: Context-Free Vision Pre-Pass
author: Sammy
description: Sends images to llama-server in an isolated context for unbiased enumeration, then injects the result as text so the main conversation can't overwrite what the model saw.
version: 0.3.0
required_open_webui_version: 0.5.0
"""

import asyncio
import hashlib
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
        api_key: str = Field(
            default="X",
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
        self, session: aiohttp.ClientSession, image_part: dict
    ) -> Optional[str]:
        """One isolated API call: image + neutral prompt, zero history."""
        key = self._digest(image_part)
        cached = self._cache_get(key)
        if cached is not None:
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
        if self.valves.id_slot >= 0:
            payload["id_slot"] = self.valves.id_slot

        try:
            async with session.post(
                self.valves.llama_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.valves.api_key}",
                    "Content-Type": "application/json",
                },
            ) as r:
                r.raise_for_status()
                data = await r.json()
            desc = data["choices"][0]["message"]["content"].strip()
            # Strip any leading chain-of-thought the model might emit
            desc = _THINK_RE.sub("", desc).strip()
            if desc:
                self._cache_put(key, desc)
            return desc or None
        except Exception as e:
            print(f"[vision-prepass] failed: {e}")
            return None

    # ---------- filter entrypoint ----------

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
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

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.valves.concurrent and len(images) > 1:
                raw = await asyncio.gather(
                    *(self._describe(session, img) for img in images),
                    return_exceptions=True,
                )
                for i, r in enumerate(raw, 1):
                    if isinstance(r, BaseException):
                        print(f"[vision-prepass] image {i} raised: {r!r}")
                raw = [r if isinstance(r, str) else None for r in raw]
            else:
                raw = [await self._describe(session, img) for img in images]

        descriptions = [
            f"{_INJECT_MARKER} {i}, produced with no "
            f"conversation context. Treat this as ground truth for what the "
            f"image contains:]\n{desc}"
            for i, desc in enumerate(raw, 1)
            if desc
        ]
        if not descriptions:
            return body  # vision pass failed; pass through untouched

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
