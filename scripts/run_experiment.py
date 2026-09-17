import json, sys, time, re
from pathlib import Path
import nbformat
from nbclient import NotebookClient

src_nb, out_nb, corpus, steps, lr, chat_prompt = sys.argv[1:7]
nb = nbformat.read(src_nb, as_version=4)

cfg = (f'CORPUS = "{corpus}"\n'
       f'CORPUS_FOLDER = "corpus"\n'
       f'TRAINING_STEPS = {steps}\n'
       f'LEARNING_RATE = {lr}\n')
nb.cells[1].source = cfg
if chat_prompt:
    nb.cells[23].source = re.sub(r'^CHAT_PROMPT = ".*?"',
                                 f'CHAT_PROMPT = "{chat_prompt}"',
                                 nb.cells[23].source, count=1, flags=re.M)

t0 = time.time()
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": str(Path(src_nb).parent)}}).execute()
nbformat.write(nb, out_nb)
print(f"executed in {time.time()-t0:.1f}s -> {out_nb}")
