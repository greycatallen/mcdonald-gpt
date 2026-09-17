#!/bin/bash
# Replays six prompts through the trained nanoGPT via a pty, so the session looks
# exactly as it does when typed by hand. See scripts/demo_chat.py.
cd "$(dirname "$0")/.."
exec ./.venv/bin/python scripts/demo_chat.py "$@"
