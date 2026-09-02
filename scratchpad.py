"""
title: Per-Chat Scratchpad
author: Sammy
description: Gives the model a private, per-chat scratchpad. The model can rewrite it directly by emitting <system_scratchpad>...</system_scratchpad> anywhere in its response — including inside its reasoning, so deductions survive after thinking is stripped from history. Optionally, a distiller side call updates the pad on turns where the model didn't write it. Notes are reinjected near the end of context on the next turn.
version: 0.3.0
required_open_webui_version: 0.5.0
"""

import asyncio
import json
import re
from collections import OrderedDict
from typing import Optional

import aiohttp
from pydantic import BaseModel, Field

# Bounded to the pad paragraph (tag line + note lines up to the next blank
# line or end) — NOT tag-to-end-of-string, other filters append text after us.
_PAD_BLOCK_RE = re.compile(
    r"\n*<system_scratchpad>.*?</system_scratchpad>", re.DOTALL | re.IGNORECASE
)
# The model's write directive. Harvested from the RAW response in outlet,
# before Open WebUI strips reasoning from history — so writes made inside
# <think>/<details type="reasoning"> blocks are captured too.
_PAD_WRITE_RE = re.compile(r"<system_scratchpad>(.*?)</system_scratchpad>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_REASONING_RE = re.compile(r'<details\s+type="reasoning".*?</details>\s*', re.DOTALL)
_PAD_DANGLING_RE = re.compile(r"<system_scratchpad>.*$", re.DOTALL | re.IGNORECASE)

class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=15,
            description="Filter execution order (lower runs first). Keep between the image filters (0/10) and the anti-sycophancy anchor (20) so the anchor stays terminal.",
        )
        model_writable: bool = Field(
            default=True,
            description="Teach the main model to rewrite the pad itself via <system_scratchpad>...</system_scratchpad>, preferably inside its reasoning. Writes replace the whole pad; an empty tag clears it.",
        )
        auto_update: bool = Field(
            default=True,
            description="On turns where the model did NOT write the pad itself, run the distiller side call. Off = pad changes only when the model writes it.",
        )
        llama_url: str = Field(
            default="http://127.0.0.1:5001/v1/chat/completions",
            description="llama-server chat completions endpoint for the side call",
        )
        llama_model: str = Field(
            default="",
            description="model to use in router mode, leave empty for the currently loaded model",
        )
        api_key: str = Field(default="", description="API key for llama-server")
        id_slot: int = Field(
            default=-1,
            description="Slot for the side call. With --parallel 2, use 1 so the main chat's cache in slot 0 survives.",
        )
        update_prompt: str = Field(
            default=(
                "You maintain a private analytical scratchpad for an ongoing "
                "conversation. Below is the current scratchpad and the latest "
                "exchange. Rewrite the scratchpad: keep entries still relevant, "
                "drop stale ones, add new deductions, contradictions noticed, "
                "open questions, and important specifics about the user or task. "
                "Be skeptical and factual, not complimentary. Telegraphic style, "
                "one line per note, max {max_lines} lines. Output ONLY the "
                "scratchpad lines, nothing else."
            ),
            description="Instruction for the side call. {max_lines} is substituted.",
        )
        max_lines: int = Field(
            default=15,
            description="Hard cap on scratchpad lines, keeps injection cheap and forces prioritization",
        )
        max_tokens: int = Field(default=400, description="Max tokens for the side call")
        temperature: float = Field(default=0.3, description="Side call temperature")
        thinking: str = Field(
            default="off",
            json_schema_extra={"enum": ["auto", "off", "on"]},
            description=(
                "Sets chat_template_kwargs.enable_thinking for the side call so "
                "thinking models don't burn max_tokens on reasoning. 'auto' = "
                "don't send the field. Needs llama-server with --jinja."
            ),
        )
        update_every_n: int = Field(
            default=1,
            description="Run the distiller every Nth assistant reply. 1 = every turn. Model-initiated writes are always applied regardless.",
        )
        max_chats: int = Field(
            default=64,
            description="How many chats' scratchpads to keep in memory (LRU)",
        )
        timeout: int = Field(default=120, description="Side call timeout in seconds")

    def __init__(self):
        self.valves = self.Valves()
        self._chats: "OrderedDict[str, dict]" = OrderedDict() # chat_id -> {"pad": str, "turns": int}

    # ---------- storage ----------

    def _get(self, chat_id: str) -> Optional[str]:
        chat = self._chats.get(chat_id)
        if chat is not None:
            self._chats.move_to_end(chat_id)
            return chat.get("pad")
        return None

    def _put(self, chat_id: str, pad: str) -> None:
        lines = [l.strip() for l in pad.splitlines() if l.strip()]
        chat = self._chats.setdefault(chat_id, {"pad": None, "turns": 0})
        
        if not lines:
            chat["pad"] = None
        else:
            chat["pad"] = "\n".join(lines[: self.valves.max_lines])
            
        self._chats.move_to_end(chat_id)
        while len(self._chats) > self.valves.max_chats:
            self._chats.popitem(last=False)

    @staticmethod
    def _chat_id(__metadata__: Optional[dict]) -> Optional[str]:
        if not __metadata__:
            return None
        cid = __metadata__.get("chat_id")
        return str(cid) if cid else None

    @staticmethod
    def _as_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""

    # ---------- inlet: inject scratchpad + write instructions ----------

    async def inlet(self, body, __user__=None, __metadata__=None) -> dict:
        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return body
        chat_id = self._chat_id(__metadata__)
        if not chat_id:
            return body

        pad = self._get(chat_id)
        if not pad and not self.valves.model_writable:
            return body

        self._strip_injected(messages)
        block = self._pad_block(pad)

        last = messages[-1]
        c = last.get("content")
        if isinstance(c, list):
            last["content"] = list(c) + [{"type": "text", "text": block}]
        else:
            last["content"] = f"{c or ''}\n\n{block}"
        return body
    
    def _pad_block(self, pad: Optional[str]) -> str:
        howto = ""
        if self.valves.model_writable:
            howto = (
                " You MUST update this pad whenever you learn new facts, task rules, or make deductions. "
                "To rewrite it, you must output exactly <system_scratchpad>your updated notes</system_scratchpad> "
                "at any point in your response, ideally inside your reasoning process. "
                "A new write completely overwrites the old pad. Limit to "
                f"{self.valves.max_lines} lines total. Outputting an empty tag clears it. "
                "The user will never see your tags."
            )
        return (
            f"<system_scratchpad>\n"
            f"PRIVATE WORKING MEMORY. The user cannot see this block. Review the current content and revise it if new information emerges.{howto}\n"
            f"Current Content:\n{pad or '(empty)'}\n"
            f"</system_scratchpad>\n"
        )

    @staticmethod
    def _strip_injected(messages: list) -> None:
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and "<system_scratchpad>" in c:
                m["content"] = _PAD_BLOCK_RE.sub("", c).rstrip()
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
                        if "<system_scratchpad>" in p["text"]:
                            p["text"] = _PAD_BLOCK_RE.sub("", p["text"]).rstrip()

    # ---------- request ----------
    
    async def request(self, body: dict, __metadata__: Optional[dict] = None) -> dict:
        """Runs before every provider call in a turn — unlike inlet(), which
        only sees the first. Without this, a tool-calling turn's follow-up
        calls lose the pad because the message list is rebuilt for each one."""
        messages = body.get("messages", [])
        if not messages:
            return body
        chat_id = self._chat_id(__metadata__)
        if not chat_id:
            return body

        pad = self._get(chat_id)
        if not pad and not self.valves.model_writable:
            return body

        self._strip_injected(messages)
        block = self._pad_block(pad)

        last = messages[-1]
        c = last.get("content")
        if isinstance(c, list):
            last["content"] = list(c) + [{"type": "text", "text": block}]
        elif isinstance(c, str):
            last["content"] = f"{c}\n\n{block}"
        else:
            # last message has no plain-text content (e.g. a bare tool_result) —
            # append a standalone note rather than guessing where to inject.
            messages.append({"role": "user", "content": block})
        return body

    # ---------- outlet: harvest model writes, else distill ----------

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __model__: Optional[str] = None,
    ) -> dict:
        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            return body
        chat_id = self._chat_id(__metadata__)
        if not chat_id:
            return body

        last = messages[-1]
        raw = self._as_text(last.get("content"))
        raw_full = raw + "\n" + self._output_texts(last)

        # 1) Model-initiated writes: harvest from the raw text, reasoning
        # included — this runs before thinking is dropped from future context,
        # which is the whole point. Last write wins.
        wrote = False
        if self.valves.model_writable:
            writes = _PAD_WRITE_RE.findall(raw_full)
            if writes:
                self._put(chat_id, writes[-1])
                wrote = True
                self._strip_writes(last)

        chat = self._chats.setdefault(chat_id, {"pad": None, "turns": 0})
        chat["turns"] += 1
        n = chat["turns"]
        self._chats.move_to_end(chat_id)

        while len(self._chats) > self.valves.max_chats:
            self._chats.popitem(last=False)

        # 2) Distiller fallback for turns without an explicit write
        if wrote or not self.valves.auto_update:
            return body
        if self.valves.update_every_n > 1 and n % self.valves.update_every_n:
            return body
        if len(messages) < 2:
            return body

        user_text = _PAD_BLOCK_RE.sub("", self._as_text(messages[-2].get("content"))).strip()
        # Distill the reply the user actually got, not the chain-of-thought
        asst_text = _REASONING_RE.sub("", raw)
        asst_text = _THINK_RE.sub("", asst_text).strip()
        current = self._get(chat_id) or "(empty)"

        prompt = self.valves.update_prompt.replace("{max_lines}", str(self.valves.max_lines))
        side_input = (
            f"{prompt}\n\nCURRENT SCRATCHPAD:\n{current}\n\n"
            f"LATEST EXCHANGE:\nUser: {user_text[:4000]}\n\n"
            f"Assistant: {asst_text[:4000]}"
        )

        if __model__ and "info" in __model__:
            model = __model__["info"].get("base_model_id") or None
        else:
            model = None

        task = asyncio.create_task(self._side_call(side_input, model))

        try:
            pad = await asyncio.wait_for(task, timeout=self.valves.timeout)
        except asyncio.TimeoutError:
            print(f"[scratchpad] side call timed out after {self.valves.timeout}s")
            task.cancel()
            return body
        except Exception as e:
            print(f"[scratchpad] side call failed: {type(e).__name__}: {e}")
            task.cancel()
            return body
        
        if pad:
            self._put(chat_id, pad)
        return body

    @staticmethod
    def _strip_writes(message: dict) -> None:
        """Remove <system_scratchpad> blocks from the reply before it is persisted."""

        def clean(text: str) -> str:
            out = _PAD_WRITE_RE.sub("", text) 
            out = _PAD_DANGLING_RE.sub("", out) 
            return re.sub(r"\n{3,}", "\n\n", out).strip()

        c = message.get("content")
        if isinstance(c, str):
            cleaned = clean(c)
            message["content"] = cleaned
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
                    p["text"] = clean(p["text"])

        for item in message.get("output") or []:
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part.get("text"), str):
                    part["text"] = clean(part["text"])

    @staticmethod
    def _output_texts(message: dict) -> str:
        """Text from message['output'] — reasoning items included. On newer
        Open WebUI this is where a write placed 'inside reasoning' actually
        ends up; it isn't inlined into the flat content string."""
        chunks = []
        for item in message.get("output") or []:
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        return "\n".join(chunks)
    # ---------- side call ----------

    async def _side_call(self, side_input: str, model: str | None = None) -> Optional[str]:
        payload = {
            "messages": [{"role": "user", "content": side_input}],
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
        elif not self.valves.llama_model and model:
            payload["model"] = model

        try:
            timeout = aiohttp.ClientTimeout(total=self.valves.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                        print(f"[scratchpad] HTTP {r.status}: {raw[:300]}")
                        return None

                    try:
                        data = json.loads(raw)
                    except Exception:
                        print(f"[scratchpad] failed to parse JSON from side call: {raw[:300]}")
                        return None
        except asyncio.TimeoutError:
            print(f"[scratchpad] side call timed out after {self.valves.timeout}s")
            return None
        except Exception as e:
            print(f"[scratchpad] update failed: {type(e).__name__}: {e}")
            return None

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        pad = (content or "").strip()
        pad = _THINK_RE.sub("", pad).strip()
        if pad.startswith("<think>"):
            pad = ""
        if not pad:
            finish = choice.get("finish_reason")
            print(
                f"[scratchpad] empty answer (finish_reason={finish}) — if the "
                f"side model thinks, set the thinking valve to 'off' (needs "
                f"--jinja) or raise max_tokens"
            )
            return None
        return pad
