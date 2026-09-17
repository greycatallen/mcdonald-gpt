"""Drive chat.py through a pseudo-terminal so the session reads like real typing.

Piping prompts into chat.py works, but the prompts never appear on screen (a pipe
is not echoed), which makes the transcript and any screenshot hard to follow. This
attaches a pty, types each prompt, and lets chat.py behave exactly as it does when
a person is at the keyboard. Nothing about the model or its replies changes.

Usage:
    python scripts/demo_chat.py [MODEL_PATH] [TRANSCRIPT_PATH]
"""
import os
import pty
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "evidence/experiment3-targeted/model.pt"
TRANSCRIPT = sys.argv[2] if len(sys.argv) > 2 else "evidence/chat/chat_transcript.json"

LONG = " ".join(["the customer reviewed the service and the product at the store today"] * 6)
PROMPTS = [
    "the customer",                                          # familiar starter pattern
    "the opposite of big is",                                # taught skill: opposites
    "the spoon is inside the drawer . the drawer contains the",  # taught skill: spatial
    "an oak is a tree . a pine is a",                        # taught skill: categories
    "quantum entanglement causes decoherence in",            # unknown words
    LONG,                                                    # exceeds the 48-token context
    "/quit",
]


def main():
    os.chdir(REPO)
    Path(TRANSCRIPT).parent.mkdir(parents=True, exist_ok=True)
    if Path(TRANSCRIPT).exists():
        os.remove(TRANSCRIPT)

    argv = [".venv/bin/python", "chat.py", "--model", MODEL, "--transcript", TRANSCRIPT]
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(argv[0], argv)

    queue = list(PROMPTS)
    buffer = b""
    while True:
        try:
            data = os.read(fd, 1024)
        except OSError:
            break
        if not data:
            break
        sys.stdout.write(data.decode(errors="replace"))
        sys.stdout.flush()
        buffer += data
        # chat.py prints "You: " and blocks; answer it.
        if buffer.rstrip().endswith(b"You:") and queue:
            time.sleep(0.4)
            os.write(fd, queue.pop(0).encode() + b"\n")
            buffer = b""
    os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
