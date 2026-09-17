"""Execute custom_llm.ipynb headlessly with a chosen configuration.

Each experiment points at its own isolated corpus folder so that one experiment's
training material can never leak into another's. This matters because
CORPUS = "classroom" means *classroom sentences PLUS every ingestible file in
CORPUS_FOLDER* -- leaving the McDonald's PDFs in the default corpus/ folder would
silently contaminate the starter baseline.

Usage:
    python scripts/run_experiment.py OUT_NOTEBOOK CORPUS_FOLDER STEPS LR [CHAT_PROMPT]

Example:
    python scripts/run_experiment.py notebooks/experiment1-starter.executed.ipynb \
        corpus_sets/starter 3000 0.001 "the customer"
"""
import argparse
import re
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_notebook")
    ap.add_argument("corpus_folder")
    ap.add_argument("steps", type=int)
    ap.add_argument("learning_rate", type=float)
    ap.add_argument("chat_prompt", nargs="?", default="the customer")
    ap.add_argument("--corpus-mode", default="classroom", choices=["classroom", "folder"])
    ap.add_argument("--source", default="custom_llm.ipynb")
    args = ap.parse_args()

    folder = Path(args.corpus_folder)
    folder.mkdir(parents=True, exist_ok=True)
    ingestible = [p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}]
    print(f"corpus folder : {folder}  ({len(ingestible)} ingestible file(s))")
    for p in sorted(ingestible):
        print(f"                - {p.name}")

    nb = nbformat.read(REPO / args.source, as_version=4)
    nb.cells[1].source = (
        f'CORPUS = "{args.corpus_mode}"\n'
        f'CORPUS_FOLDER = "{args.corpus_folder}"\n'
        f'TRAINING_STEPS = {args.steps}\n'
        f'LEARNING_RATE = {args.learning_rate}\n'
    )
    nb.cells[23].source = re.sub(
        r'^CHAT_PROMPT = ".*?"',
        f'CHAT_PROMPT = "{args.chat_prompt}"',
        nb.cells[23].source, count=1, flags=re.M,
    )

    started = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(REPO)}}).execute()

    out = Path(args.out_notebook)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(f"executed in {time.time() - started:.1f}s -> {out}")


if __name__ == "__main__":
    main()
