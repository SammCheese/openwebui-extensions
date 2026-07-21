# openwebui-extensions

Filter functions for [Open WebUI](https://github.com/open-webui/open-webui), built for a local llama.cpp setup.

> ## ⚠️ Disclaimer
>
> **These filters were AI-generated (with Claude) for personal use.**
>
> They are tuned to one specific homelab: a llama-server instance, small local
> instruct models (Gemma-class), and a self-hosted "Open Terminal" shell API.
> They are **not** audited for security, not tested beyond that environment,
> and several of them deliberately inject text into your prompts and rewrite
> model responses. Read the code before installing any of them, change the
> default URLs/tokens, and use at your own risk.

## The filters

All four are Open WebUI **filter functions** (Admin Panel → Functions → paste the file contents). They coordinate through a `priority` valve — lower runs first, and the order matters because they all append text to the same user message:

| Priority | File                           | What it does                                        |
| -------- | ------------------------------ | --------------------------------------------------- |
| 0        | `add_image_to_openterminal.py` | Mirrors chat images into a terminal filesystem      |
| 10       | `nocontextimagepass.py`        | Context-free vision pass over attached images       |
| 15       | `scratchpad.py`                | Private per-chat scratchpad the model can write to  |
| 20       | `antisycophancy.py`            | Anti-sycophancy conduct anchor + response scrubbing |

The anti-sycophancy anchor must stay last so its reminder is the final text the model sees; the image mirror must run first so it still sees images even if the vision pass is configured to strip them.

### `add_image_to_openterminal.py` — Image → Open Terminal Mirror

Saves every image attached to a user message into an Open Terminal filesystem (base64 upload + `base64 -d` decode) and appends a note with the saved path(s) to the message, so a shell-capable model can operate on the files. Content-addressed filenames (SHA-256 digest) plus an in-memory LRU cache and an optional remote `test -f` check mean retries, edits, and regenerations never re-upload or duplicate files. Per-chat subdirectories keep chats separate; chat ids are sanitized before touching the shell. Emits status lines in the chat UI while uploading. Does nothing until the `bearer_token` valve is set.

### `nocontextimagepass.py` — Context-Free Vision Pre-Pass

Sends each attached image to llama-server in an isolated request with a neutral enumeration prompt — zero conversation history — and injects the result into the message as an "independent visual inventory". The point: the main conversation can't bias or overwrite what the model saw. Handles thinking-capable vision models (a `thinking` valve sets `chat_template_kwargs.enable_thinking`, needs `--jinja`), parses all the response shapes llama-server produces (`null` content, reasoning fields, list parts), caches descriptions by image digest so regenerations don't re-run inference, and reports failures with the actual cause in the chat status line. `keep_image` off makes the pass fully context-proof by removing the image entirely — if you do that, anything that needs the image must run at a lower priority.

### `scratchpad.py` — Per-Chat Scratchpad

Gives the model private working memory that survives across turns without living in visible chat history. The model rewrites it by emitting `<scratchpad>...</scratchpad>` anywhere in its output — **including inside its reasoning block**, which is the interesting part: deductions made while thinking get captured by the outlet before Open WebUI strips reasoning from future context. Writes replace the whole pad, an empty tag clears it, and the tag is stripped before the reply is stored. On turns where the model doesn't write, an optional distiller side call (small llama-server request) condenses the latest exchange into the pad instead. The pad is reinjected near the end of context each turn. Pads are in-memory only: a server restart loses them.

### `antisycophancy.py` — Anti-Sycophancy Anchor

Counters positivity/confirmation bias in small instruct models whose system-prompt adherence decays with context length. Three mechanisms, each toggleable: re-injects a condensed conduct reminder at the very end of context every turn (recency beats a decaying system prompt); optionally adds `logit_bias` against praise-opener tokens (tokenizer-specific, off by default); and an outlet scrubber that deletes leading "You're absolutely right!"-style sentences from responses. The scrubbed version is what gets persisted, and scrubbing is idempotent, so regenerations behave. Note the scrubber removes whole sentences — a factual claim inside a flattering opener goes with it.

## Shared behavior

- **Retry/regeneration safe**: inlet injections are never persisted by Open WebUI, so every filter re-runs on regeneration against clean history; digest caches (images) and marker checks (injected blocks) prevent duplicate work and stacked injections.
- **Fail open**: every network call is wrapped; on any failure the filters log, optionally surface the reason as a chat status line, and pass the request through untouched.
- **In-memory state only**: dedupe caches, scratchpads, and turn counters live in the filter process. Restarting Open WebUI resets them (the image mirror re-checks the remote filesystem, so it recovers; scratchpads don't).

## Requirements

- Open WebUI ≥ 0.5.0 (`aiohttp` and `pydantic` ship with it)
- A llama.cpp `llama-server` for the vision pass and scratchpad distiller (run with `--jinja` if you use the `thinking` valves)
- An Open Terminal instance for the image mirror
