"""
title: Image → Open Terminal Mirror
author: Sammy
description: Saves every image attached to a chat message into the Open Terminal filesystem (via base64 text write + decode), and injects the file path into the message so the model can operate on it with shell tools. Async, with content-hash dedupe so retries/edits don't re-upload.
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

_NOTE_PREFIX = "[Attached image file(s) are also saved in the terminal filesystem"


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter execution order (lower runs first). Keep this LOWEST so images are mirrored before the vision pre-pass can strip them (keep_image=False) and before the anti-sycophancy anchor is appended.",
        )
        terminal_url: str = Field(
            default="http://127.0.0.1:8002",
            description="Open Terminal base URL (no trailing slash)",
        )
        bearer_token: str = Field(
            default="",
            description="Bearer token for Open Terminal",
        )
        dest_dir: str = Field(
            default="chat_images",
            description="Directory (relative to Open Terminal's working dir) where images land",
        )
        per_chat_subdir: bool = Field(
            default=True,
            description="Create a subdirectory per chat id, so images from different chats don't mix",
        )
        inject_note: bool = Field(
            default=True,
            description="Append a text note with the saved path(s) to the user message so the model knows the file exists",
        )
        check_remote: bool = Field(
            default=True,
            description="Before uploading, run a cheap `test -f` on the terminal to see if the file already exists (survives filter reloads, unlike the in-memory cache)",
        )
        exec_wait: float = Field(
            default=20.0,
            description="Seconds to wait for the base64-decode command to finish",
        )
        max_bytes: int = Field(
            default=15_000_000,
            description="Skip images whose base64 payload exceeds this size",
        )
        timeout: int = Field(default=60, description="HTTP timeout per request")
        cache_size: int = Field(
            default=256,
            description="How many digest→path entries to keep in the in-memory dedupe cache",
        )

    def __init__(self):
        self.valves = self.Valves()
        # digest -> saved path. Survives across turns within one server process,
        # so a retry/edit of a message with the same image is a cache hit.
        self._saved: "OrderedDict[str, str]" = OrderedDict()

    # ---------- dedupe cache ----------

    def _cache_get(self, digest: str) -> Optional[str]:
        path = self._saved.get(digest)
        if path is not None:
            self._saved.move_to_end(digest)
        return path

    def _cache_put(self, digest: str, path: str) -> None:
        self._saved[digest] = path
        self._saved.move_to_end(digest)
        while len(self._saved) > self.valves.cache_size:
            self._saved.popitem(last=False)

    # ---------- Open Terminal helpers ----------

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.valves.bearer_token}",
            "Content-Type": "application/json",
        }

    async def _write_text(
        self, session: aiohttp.ClientSession, path: str, content: str
    ) -> bool:
        try:
            async with session.post(
                f"{self.valves.terminal_url}/files/write",
                json={"path": path, "content": content},
                headers=self._headers(),
            ) as r:
                return r.ok
        except Exception as e:
            print(f"[img-mirror] write error: {e}")
            return False

    async def _exec(
        self, session: aiohttp.ClientSession, command: str, wait: Optional[float] = None
    ) -> Optional[dict]:
        wait = self.valves.exec_wait if wait is None else wait
        try:
            async with session.post(
                f"{self.valves.terminal_url}/execute",
                params={"wait": wait, "tail": 5},
                json={"command": command},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=self.valves.timeout + wait),
            ) as r:
                return await r.json() if r.ok else None
        except Exception as e:
            print(f"[img-mirror] exec error: {e}")
            return None

    # ---------- image handling ----------

    _DATA_URL = re.compile(
        r"^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<b64>.+)$", re.DOTALL
    )

    async def _save_image(
        self, session: aiohttp.ClientSession, data_url: str, dest_dir: str, index: int
    ) -> Optional[str]:
        m = self._DATA_URL.match(data_url.strip())
        if not m:
            return None  # remote URL or unexpected format; skip
        b64 = m.group("b64")
        if len(b64) > self.valves.max_bytes:
            print(f"[img-mirror] image {index} exceeds max_bytes, skipped")
            return None

        ext = m.group("ext").lower()
        ext = {"jpeg": "jpg", "svg+xml": "svg"}.get(ext, ext)

        # Deterministic, content-addressed filename: same image → same path,
        # so a retry can never create a duplicate file even on cache miss.
        digest = hashlib.sha256(b64.encode("ascii", "ignore")).hexdigest()[:16]
        img_path = f"{dest_dir}/img_{digest}.{ext}"
        b64_path = f"{dest_dir}/img_{digest}.b64"

        # 1) In-memory dedupe (fast path for retries/edits/regenerations)
        cached = self._cache_get(digest)
        if cached:
            return cached

        # 2) Remote dedupe (survives filter reload / server restart)
        if self.valves.check_remote:
            result = await self._exec(
                session, f"test -f '{img_path}' && echo EXISTS", wait=5.0
            )
            if result and "EXISTS" in str(result):
                self._cache_put(digest, img_path)
                return img_path

        # 3) Upload base64 as text (parent dirs auto-created)
        if not await self._write_text(session, b64_path, b64):
            print(f"[img-mirror] write failed for image {index}")
            return None

        # 4) Decode to binary, verify, clean up intermediate
        result = await self._exec(
            session,
            f"base64 -d '{b64_path}' > '{img_path}' && rm '{b64_path}' "
            f"&& ls -la '{img_path}'",
        )
        if not result:
            print(f"[img-mirror] decode exec failed for image {index}")
            return None

        self._cache_put(digest, img_path)
        return img_path

    # ---------- filter entrypoint ----------

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        if not self.valves.bearer_token:
            return body  # unconfigured; pass through silently

        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return body

        last = messages[-1]
        content = last.get("content")
        if not isinstance(content, list):
            return body

        # If our note is already present (edited message that kept the injected
        # text), don't process again — otherwise we'd stack duplicate notes.
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and _NOTE_PREFIX in (part.get("text") or "")
            ):
                return body

        images = [
            p for p in content if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        if not images:
            return body

        dest = self.valves.dest_dir
        if self.valves.per_chat_subdir and __metadata__:
            # chat_id lands inside shell commands; allow only filename-safe chars
            chat_id = re.sub(
                r"[^A-Za-z0-9_-]", "_", str(__metadata__.get("chat_id") or "no_chat")
            )[:32]
            dest = f"{dest}/{chat_id}"

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(
                *(
                    self._save_image(
                        session, (p.get("image_url") or {}).get("url", ""), dest, i
                    )
                    for i, p in enumerate(images, 1)
                ),
                return_exceptions=True,
            )

        saved = []
        for i, r in enumerate(results, 1):
            if isinstance(r, str):
                saved.append(r)
            elif isinstance(r, BaseException):
                print(f"[img-mirror] image {i} raised: {r!r}")

        if saved and self.valves.inject_note:
            note = (
                _NOTE_PREFIX
                + " and can be operated on with shell commands: "
                + ", ".join(saved)
                + "]"
            )
            last["content"] = list(content) + [{"type": "text", "text": note}]

        return body
