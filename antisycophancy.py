"""
title: Anti-Sycophancy Anchor
author: Sammy
description: Counters positivity/confirmation bias in small instruct models (Gemma etc.) whose system prompt attention decays with context. Re-injects a condensed rule block at the END of context every turn, optionally logit-biases sycophantic opener tokens, and scrubs agreement-openers from responses.
version: 0.2.1
required_open_webui_version: 0.5.0
"""

import json
import re
from typing import Optional

from pydantic import BaseModel, Field

_ANCHOR_TAG = "[conduct reminder"
# Matches only the anchor paragraph: the bracketed tag plus text up to the
# next blank line or end of string. Deliberately NOT "tag to end of string" —
# other filters may append their own text after our anchor, and that must
# survive the next turn's cleanup.
_ANCHOR_BLOCK = re.compile(
    r"\n*\[conduct reminder[^\]]*\].*?(?=\n\n|\Z)", re.DOTALL
)


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

    # ---------- inlet: recency reinjection ----------

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return body
        if len(messages) < self.valves.min_messages:
            return body

        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if self.valves.every_n_turns > 1 and user_turns % self.valves.every_n_turns:
            return body

        # Remove any anchor we injected on earlier turns so exactly one copy
        # exists, always at the end. Stale copies mid-context waste attention.
        for m in messages:
            c = m.get("content")
            if isinstance(c, str) and _ANCHOR_TAG in c:
                m["content"] = _ANCHOR_BLOCK.sub("", c).rstrip()
            elif isinstance(c, list):
                m["content"] = [
                    p
                    for p in c
                    if not (
                        isinstance(p, dict)
                        and p.get("type") == "text"
                        and _ANCHOR_TAG in (p.get("text") or "")
                    )
                ]
        body["messages"] = [
            m
            for m in messages
            if not (m.get("role") == "system" and _ANCHOR_TAG in str(m.get("content")))
        ]
        messages = body["messages"]

        anchor = f"{_ANCHOR_TAG}, not part of the user's message:]\n{self.valves.reminder_text}"

        if self.valves.as_system_message:
            messages.insert(-1, {"role": "system", "content": anchor})
        else:
            last = messages[-1]
            c = last.get("content")
            if isinstance(c, list):
                last["content"] = list(c) + [{"type": "text", "text": anchor}]
            else:
                last["content"] = f"{c}\n\n{anchor}"

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
        if not isinstance(text, str) or not text:
            return body

        try:
            pattern = re.compile(self.valves.opener_patterns, re.IGNORECASE)
        except re.error as e:
            print(f"[anti-syco] bad opener_patterns regex: {e}")
            return body

        removed = 0
        while removed < self.valves.max_scrub_sentences:
            # First sentence = up to first ., !, or ? followed by space/newline
            m = re.match(r"\s*(.+?[.!?])(\s+|$)", text, re.DOTALL)
            if not m:
                break
            sentence = m.group(1)
            if pattern.search(sentence):
                text = text[m.end() :]
                removed += 1
            else:
                break

        if removed:
            text = text.lstrip()
            if text:
                text = text[0].upper() + text[1:]
                last["content"] = text
            # If scrubbing emptied the response, leave the original alone
        return body
