"""
title: Context-Free Vision Pre-Pass
author: Sammy
description: Sends images to llama-server in an isolated context for unbiased enumeration, then injects the result as text so the main conversation can't overwrite what the model saw.
version: 0.7.5
required_open_webui_version: 0.5.0
"""

import asyncio
import hashlib
import json
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field

_INJECT_MARKER = "[Independent visual inventory of image"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NDAgNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDcuMy4xIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjYgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTE5MiA2NGMwLTM1LjMgMjguNy02NCA2NC02NEw1NzYgMGMzNS4zIDAgNjQgMjguNyA2NCA2NGwwIDIyNGMwIDM1LjMtMjguNyA2NC02NCA2NGwtMzIwIDBjLTM1LjMgMC02NC0yOC43LTY0LTY0bDAtMjI0ek0zMjAgOTZhMzIgMzIgMCAxIDAgLTY0IDAgMzIgMzIgMCAxIDAgNjQgMHptMTU2LjUgMTEuNUM0NzIuMSAxMDAuNCA0NjQuNCA5NiA0NTYgOTZzLTE2LjEgNC40LTIwLjUgMTEuNWwtNTQgODguMy0xNy45LTI1LjZjLTQuNS02LjQtMTEuOC0xMC4yLTE5LjctMTAuMnMtMTUuMiAzLjgtMTkuNyAxMC4ybC01NiA4MGMtNS4xIDcuMy01LjggMTYuOS0xLjYgMjQuOFMyNzkuMSAyODggMjg4IDI4OGwyNTYgMGM4LjcgMCAxNi43LTQuNyAyMC45LTEyLjNzNC4xLTE2LjgtLjUtMjQuM2wtODgtMTQ0ek0xNDQgMTI4bDAgMTYwYzAgNjEuOSA1MC4xIDExMiAxMTIgMTEybDE5MiAwIDAgMTZjMCAzNS4zLTI4LjcgNjQtNjQgNjRMNjQgNDgwYy0zNS4zIDAtNjQtMjguNy02NC02NEwwIDE5MmMwLTM1LjMgMjguNy02NCA2NC02NGw4MCAwek01MiAxOTZsMCAyNGMwIDguOCA3LjIgMTYgMTYgMTZsMjQgMGM4LjggMCAxNi03LjIgMTYtMTZsMC0yNGMwLTguOC03LjItMTYtMTYtMTZsLTI0IDBjLTguOCAwLTE2IDcuMi0xNiAxNnptMTYgODBjLTguOCAwLTE2IDcuMi0xNiAxNmwwIDI0YzAgOC44IDcuMiAxNiAxNiAxNmwyNCAwYzguOCAwIDE2LTcuMiAxNi0xNmwwLTI0YzAtOC44LTcuMi0xNi0xNi0xNmwtMjQgMHptMCA5NmMtOC44IDAtMTYgNy4yLTE2IDE2bDAgMjRjMCA4LjggNy4yIDE2IDE2IDE2bDI0IDBjOC44IDAgMTYtNy4yIDE2LTE2bDAtMjRjMC04LjgtNy4yLTE2LTE2LTE2bC0yNCAweiIvPjwvc3ZnPg=="

class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=10,
            description="Filter execution order (lower runs first). Keep this between the terminal mirror (0) and the anti-sycophancy anchor (20).",
        )
        ENABLED: bool = Field(
            default=True,
            description="If false, the vision pass is skipped entirely.",
        )
        REGENERATE: bool = Field(
            default=False,
            description="If true, the vision pass is re-run on every retry/regeneration. If false, the pass is only run once per image digest and cached for subsequent retries.",
        )
        llama_url: str = Field(
            default="http://127.0.0.1:5001/v1",
            description="llama-server chat completions endpoint (e.g. http://127.0.0.1:5001/v1)",
        )
        LLAMA_MODEL: Optional[str] = Field(
            default=None,
            description="model to use in router mode, leave empty for the currently loaded model",
        )
        API_KEY: Optional[str] = Field(
            default=None,
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

    class UserValves(BaseModel):
        REGENERATE: bool = Field(
            default=False,
            description="Override the admin default: If true, the vision pass is re-run on every retry/regeneration. If false, the pass is only run once per image digest and cached for subsequent retries.",
        )
        ENABLED: bool = Field(
            default=True,
            description="If false, the vision pass is skipped entirely.",
        )
        FOCUS_PROMPT: str = Field(
            default="",
            description="If non-empty, this string is appended to the enumeration prompt for this user. Use it to bias the enumeration toward certain details (e.g. 'Focus on anatomical details and text in the image').",
        )

    def __init__(self):
        self.valves = self.Valves()
        # image digest -> description. Retrying/regenerating a turn resends the
        # same image; the digest matches and we skip the API call entirely.
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        # Last failure reason, surfaced in the UI status so users don't have
        # to tail the server log to find out why the pass produced nothing.
        self._last_error: Optional[str] = None
        self.toggle = (
            True  # user-controllable chip; clicking it opens the UserValves modal below
        )
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

    def _clean_model_id(self, model_id: Optional[str]) -> Optional[str]:
        """Clean up the model ID to avoid sending empty strings or whitespace."""
        if model_id is None:
            return None
        cleaned = model_id.strip().strip('"').strip("'")
        return cleaned if cleaned else None

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
        self,
        session: aiohttp.ClientSession,
        image_part: dict,
        model: Optional[str] = None,
        regenerate: Optional[bool] = False,
        focus_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """One isolated API call: image + neutral prompt, zero history."""
        key = self._digest(image_part)
        cached = self._cache_get(key)

        if cached is not None and not regenerate:
            return cached

        if focus_prompt is not None:
            prompt = f"{self.valves.enumeration_prompt}\n\n{focus_prompt.strip()}"
        else:
            prompt = self.valves.enumeration_prompt

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_part,
                        {"type": "text", "text": prompt},
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

        if model is not None:
            payload["model"] = model

        try:
            url = self.valves.llama_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}

            if self.valves.API_KEY:
                headers["Authorization"] = f"Bearer {self.valves.API_KEY}"

            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.valves.timeout,
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

        if not regenerate:
            self._cache_put(key, desc)

        return desc

    def _fail(self, reason: str) -> None:
        self._last_error = reason
        print(f"[vision-prepass] failed: {reason}")

    def _strip_images(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove all image payloads from messages so the base model never sees them."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Keep only text blocks, drop image_url blocks
                text_only = [p for p in content if p.get("type") != "image_url"]
                # Flatten to plain string if just text remains
                msg["content"] = (
                    " ".join(p.get("text", "") for p in text_only).strip()
                    if text_only
                    else ""
                )
            # Drop Ollama-style images array entirely
            if "images" in msg:
                del msg["images"]
        return messages

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
        user_valves = self.user_valves(__user__)

        enabled = user_valves.get("ENABLED", self.valves.ENABLED)
        regenerate = user_valves.get("REGENERATE", self.valves.REGENERATE)
        focus_prompt = user_valves.get("FOCUS_PROMPT", "")

        if not enabled:
            return body

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

        model_name = (
            self._clean_model_id(self.valves.LLAMA_MODEL)
            or self._clean_model_id(body.get("model"))
            or None
        )

        self._last_error = None

        # On a retry/regeneration every digest hits the cache, so don't show an
        # "analyzing" status for work that won't happen.
        misses = sum(
            1
            for img in images
            if self._cache_get(self._digest(img)) is None and not regenerate
        )

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.valves.concurrent and len(images) > 1:
                if misses:
                    await self._status(
                        __event_emitter__,
                        f"Running context-free vision pass on {misses} image(s)…",
                    )
                raw = await asyncio.gather(
                    *(self._describe(session, img, model=model_name, regenerate=regenerate, focus_prompt=focus_prompt) for img in images),
                    return_exceptions=True,
                )
                for i, r in enumerate(raw, 1):
                    if isinstance(r, BaseException):
                        print(f"[vision-prepass] image {i} raised: {r!r}")
                raw = [r if isinstance(r, str) else None for r in raw]
            else:
                raw = []
                for i, img in enumerate(images, 1):
                    if self._cache_get(self._digest(img)) is None or regenerate:
                        await self._status(
                            __event_emitter__,
                            f"Context-free vision pass: image {i}/{len(images)}…",
                        )
                    raw.append(
                        await self._describe(
                            session, img, model=model_name, regenerate=regenerate, focus_prompt=focus_prompt
                        )
                    )

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

        if not self.valves.keep_image:
            # Strip all images from the conversation so the base model can't see them.
            messages = self._strip_images(messages)
            body["messages"] = messages

        return body
