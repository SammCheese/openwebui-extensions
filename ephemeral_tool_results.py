"""
title: Ephemeral Tool Results
author: Sammy
version: 0.1.0
license: MIT
required_open_webui_version: 0.7.0
description: Let selected tool results live for exactly one turn, then drop them from the payload sent to the model.

Pairs with the Sub Agent tool. The sub-agent's return value is only ever read by
the main agent, but Open WebUI persists it as a role="tool" message and replays
it on every subsequent request. This filter keeps the result for the turn that
produced it and redacts it afterwards.

Boundary rule: everything at or after the LAST user message is the turn in
flight and is left untouched. Everything before it is history and is eligible
for redaction. This is what makes the filter safe when the native
function-calling follow-up request re-enters the inlet chain.
"""

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """Flatten string or list-of-parts content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter ordering. Only matters if another filter also rewrites message history.",
        )
        ENABLED: bool = Field(
            default=True,
            description="Master switch.",
        )
        TOOL_NAMES: str = Field(
            default="run_sub_agent,run_parallel_sub_agents",
            description=(
                "Comma-separated tool function names whose results are ephemeral. "
                "Matched via tool_call_id against the preceding assistant message, "
                "so no changes to the tool source are needed."
            ),
        )
        SENTINEL: str = Field(
            default="",
            description=(
                "Optional. If set, a tool result ALSO matches when its content contains "
                'this substring (e.g. "_ephemeral_subagent"). Useful when you control the '
                "tool's return value and want matching that survives tool renames."
            ),
        )
        MODE: Literal["redact", "excise"] = Field(
            default="redact",
            description=(
                "redact: keep the tool message, replace its content with STUB_TEXT. Safe — "
                "tool_call pairing stays valid. "
                "excise: remove the tool message and prune the matching entry from the "
                "assistant's tool_calls. Saves a few more tokens, more ways to go wrong."
            ),
        )
        STUB_TEXT: str = Field(
            default="[Result consumed in a previous turn and discarded to save context. "
            "If you need this information again, re-run the tool.]",
            description="Replacement content in redact mode. Set to an empty string to blank it entirely.",
        )
        SKIP_SUB_AGENT_REQUESTS: bool = Field(
            default=True,
            description=(
                "Leave requests originating from inside the sub-agent loop alone "
                "(metadata.task == 'sub_agent'). Turning this off will corrupt the "
                "sub-agent's own iteration history. Keep it on."
            ),
        )
        DEBUG: bool = Field(
            default=False,
            description="Log how many results were dropped per request.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------

    def _ephemeral_tool_names(self) -> set[str]:
        return {
            name.strip() for name in self.valves.TOOL_NAMES.split(",") if name.strip()
        }

    def _build_id_to_name(self, messages: list) -> dict[str, str]:
        """Map tool_call_id -> function name using assistant messages."""
        mapping: dict[str, str] = {}
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                func = call.get("function")
                if isinstance(call_id, str) and isinstance(func, dict):
                    name = func.get("name")
                    if isinstance(name, str):
                        mapping[call_id] = name
        return mapping

    def _is_ephemeral(
        self, msg: dict, id_to_name: dict[str, str], names: set[str]
    ) -> bool:
        call_id = msg.get("tool_call_id")
        if isinstance(call_id, str) and id_to_name.get(call_id) in names:
            return True
        sentinel = self.valves.SENTINEL.strip()
        if sentinel and sentinel in _content_to_text(msg.get("content")):
            return True
        return False

    # ------------------------------------------------------------------

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.ENABLED or not isinstance(body, dict):
            return body

        metadata = body.get("metadata")
        if (
            self.valves.SKIP_SUB_AGENT_REQUESTS
            and isinstance(metadata, dict)
            and metadata.get("task") == "sub_agent"
        ):
            return body

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body

        names = self._ephemeral_tool_names()
        if not names and not self.valves.SENTINEL.strip():
            return body

        # Anything from this index onward belongs to the turn in flight.
        cutoff = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "user":
                cutoff = i
                break

        id_to_name = self._build_id_to_name(messages)
        excised_ids: set[str] = set()
        result: list = []
        hits = 0

        for i, msg in enumerate(messages):
            if (
                i >= cutoff
                or not isinstance(msg, dict)
                or msg.get("role") != "tool"
                or not self._is_ephemeral(msg, id_to_name, names)
            ):
                result.append(msg)
                continue

            hits += 1
            if self.valves.MODE == "redact":
                result.append({**msg, "content": self.valves.STUB_TEXT})
            else:
                call_id = msg.get("tool_call_id")
                if isinstance(call_id, str):
                    excised_ids.add(call_id)
                # message dropped

        # In excise mode, prune the now-orphaned tool_calls entries so the
        # provider doesn't reject the request.
        if excised_ids:
            pruned: list = []
            for msg in result:
                if (
                    not isinstance(msg, dict)
                    or msg.get("role") != "assistant"
                    or not msg.get("tool_calls")
                ):
                    pruned.append(msg)
                    continue

                kept = [
                    call
                    for call in msg["tool_calls"]
                    if not (isinstance(call, dict) and call.get("id") in excised_ids)
                ]
                if kept:
                    pruned.append({**msg, "tool_calls": kept})
                    continue

                stripped = {k: v for k, v in msg.items() if k != "tool_calls"}
                if _content_to_text(stripped.get("content")).strip():
                    pruned.append(stripped)
                # otherwise the assistant turn carried nothing but the call: drop it
            result = pruned

        if hits:
            body["messages"] = result
            if self.valves.DEBUG:
                log.info(
                    f"[EphemeralToolResults] {self.valves.MODE}ed {hits} tool "
                    f"result(s) before message index {cutoff}"
                )

        return body
