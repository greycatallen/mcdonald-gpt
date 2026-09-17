# McDonald GPT — Class 4: Building a Custom LLM

Training a 112,000-parameter nanoGPT from scratch on a classroom corpus, then measuring
it against a fixed 48-case language eval suite before and after training — twice, once on
the starter corpus and once on an extended corpus.

**Repository:** https://github.com/greycatallen/mcdonald-gpt
**Model:** Andrej Karpathy's nanoGPT (`nanogpt_model.py`, pinned to upstream commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, SHA-256 verified by the notebook at runtime).
No API keys, no pretrained weights, no other model. CPU only.

> ### ⚠️ Status: IN PROGRESS
>
> Every section below marked **`⬜ PENDING`** is waiting on a run that has not happened yet.
> Nothing in this README is estimated, predicted-as-actual, or copied from the reference
> run shipped in the upstream repo. When a number appears here, it came out of a run in
> this repository and is linked to the file it came from.
>
> | Stage | State |
> |---|---|
> | Pipeline smoke test (10 steps) | ✅ done — [`evidence/smoke-10-steps/`](evidence/smoke-10-steps/) |
> | Prediction written and locked | ⬜ pending — being revised, must be committed *before* Experiment 1 |
> | Experiment 1: starter corpus | ⬜ pending |
> | Experiment 2: extended corpus | ⬜ pending |
> | Chat transcript (3+ real interactions) | ⬜ pending |
> | Conceptual write-ups (in my own words) | ⬜ pending |

---

## 1. Overview

This is the Class 4 assignment: choose a corpus, training steps, and a learning rate; train
a tiny word-token transformer; test it on an eval suite that is deliberately kept out of the
training data; and talk to the result through a chat interface.

The model is small enough to inspect completely. Two transformer blocks, four attention
heads, 64-dimensional embeddings, a 48-token context window, trained with AdamW using
warmup and cosine decay. At the starter vocabulary size of 136 word types that is
**111,872 parameters** — roughly one millionth of a frontier model.

The point is not to build a capable chatbot. It is to be able to point at a specific token
ID, a specific 64-number vector, a specific gradient, and a specific weight update, and
explain how they connect.

---

## 2. How to open and run

### Local (what was actually used here)

```bash
git clone https://github.com/greycatallen/mcdonald-gpt.git
cd mcdonald-gpt
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt numpy
```

`numpy` is installed explicitly: `run_evals.py` hashes model weights via
`tensor.numpy()`, which fails with `RuntimeError: Numpy is not available` on a
torch-only environment. This is the one setup deviation from the upstream instructions.

Then open `custom_llm.ipynb` in Jupyter or VS Code against `.venv`, set the three choices
in Section 1, and Run All.

### Google Colab

Open `custom_llm.ipynb` in Colab; the default CPU runtime is enough. Run Sections 1–2 once
to create `/content/corpus`, upload extension files into that folder via the Files sidebar,
then Run All. Opening a notebook from GitHub does **not** copy this repository's folders.

### Reproducing a run headlessly

[`scripts/run_experiment.py`](scripts/run_experiment.py) sets the three choices and executes
the notebook with outputs preserved:

```bash
./.venv/bin/python scripts/run_experiment.py custom_llm.ipynb notebooks/out.ipynb classroom 3000 0.001 "the customer"
```

### Verifying the eval suite was never modified

```bash
shasum -a 256 evals/language_evals.json
# e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7
```

This matches the hash pinned inside the notebook's own setup cell and the upstream
repository, so the 48 cases are provably unchanged across every run below.

---

## 3. My three choices and my reasons

| Choice | Value | Reason |
|---|---|---|
| Corpus | Experiment 1: `classroom` (starter only) · Experiment 2: `classroom` + McDonald's financial statements and CEO letters · Experiment 3: `classroom` + material targeted at ≥2 eval categories | Three runs, because "does adding data help?" and "does adding *task-matched* data help?" are different questions and a single extension run cannot separate them. |
| Training steps | 3000 | The notebook's designated main-experiment budget (`# 10 for setup; 3000 for the main experiment`) ≈ 23 passes over the corpus at batch size 32. A 10-step smoke test was run first and is reported below as a null result. Runtime is ~8 s, so the budget was not chosen to save time — it was chosen to be comparable to the reference configuration. |
| Learning rate | 0.001 | The standard AdamW default and well-matched to a 112k-parameter model. Held **identical across all three experiments** so that any difference between them is attributable to the corpus, not the optimizer. |

**On the learning rate specifically** — why too large or too small is a problem: the learning
rate scales every weight update. Too large and the optimizer overshoots the minimum it is
descending toward; the loss curve goes jagged or diverges outright. Too small and each update
barely moves the weights, so the model is still underfit when the step budget runs out. The
notebook applies warmup then cosine decay on top of the value set here, so this is the
*peak* rate, not a constant one.

### Pipeline smoke test (10 steps) — a real, documented null result

Before the graded run, the pipeline was validated end-to-end at `TRAINING_STEPS = 10`, the
value the notebook labels `# 10 for setup`. The whole notebook completed in **7.9 seconds**
with no errors. It also learned essentially nothing, which is the expected and correct
outcome:

| Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable | Coverage |
|---|---|---|---|---|
| Untrained | 9 | 24 | 37.5% | 50% |
| Trained (10 steps) | 9 | 24 | 37.5% | 50% |

Loss moved 4.926 → 4.208. Uniform-guess loss for a 136-word vocabulary is
ln(136) = **4.913**, so after 10 steps the model was still close to a random guesser. The
overall eval score did not move at all; the only change was noise redistributing between
groups (`starter_patterns` 6→5, `starter_transfer` 3→4).

This run is kept deliberately, as evidence that the harness works and that a 10-step budget
is not a training budget. Full output: [`evidence/smoke-10-steps/`](evidence/smoke-10-steps/).

---

## 4. My prediction, written before training

> Committed in this repository **before any graded run**, and not edited afterward. Git
> history is the timestamp: this section was committed at the tip named in §6, before the
> Experiment 1 commit that follows it.

### My prediction (Allen)

**P1. Adding new training material will significantly improve the evaluation results.**

**P2. Training on McDonald's financial statements and CEO letters to consumers will be
enough, on its own, to produce that significant improvement.**

My reasoning: the starter corpus is small and narrow. Real business prose is far larger and
more varied than synthetic classroom sentences, so a model trained on it should have more
language to draw on and should therefore do better on a general language test.

**What counts as "significant", agreed before the run:** an increase of **at least 5 cases
out of 48** in all-case success over the Experiment 1 trained baseline. Anything smaller is
noise at this scale — a single case is 2.1 percentage points.

### Recorded disagreement (assistant), before the run

I expect **P2 to be falsified**, and I am recording why in advance so the comparison is
honest rather than reconstructed afterward:

1. The 24 `extend_corpus` cases need words like `salmon`, `robin`, `freezes`, `opposite`,
   `lent`, `thanked`, `cold`, `ice`, `blue`. None of these occur in a 10-K or a shareholder
   letter, so those cases should stay **out of vocabulary and score 0**.
2. The tokenizer keeps only the **509 most frequent training token types**. Financial
   vocabulary competes for those slots, so adding a large off-topic corpus can *evict*
   starter words and **reduce** coverage — making the score go **down**, not up.
3. Therefore I expect Experiment 3 (task-matched material) to beat Experiment 2
   (more, off-topic material), even though Experiment 2 adds far more text.

If P1/P2 hold and this is wrong, that is the more interesting result and it gets reported
as plainly as the reverse.

### Shared predictions for Experiment 1 (starter corpus)

| # | Claim | Committed value |
|---|---|---|
| 1 | Validation loss falls from ~4.93 | to **1.6 – 2.2** |
| 2 | Train/validation gap stays small — **no real overfitting**, because both splits come from the same template generator | gap **< 0.2** |
| 3 | `extend_corpus` cannot improve — the words are absent from the vocabulary | **0 / 24, 0% coverage** |
| 4 | `starter_patterns` improves substantially | 37.5% → **60 – 85%** |
| 5 | `starter_transfer` improves less, since it reuses known words in new sentence shapes | 37.5% → **40 – 60%** |
| 6 | Overall, hard-capped at 24/48 by out-of-vocabulary cases | **14 – 19 / 48** |
| 7 | `customer`'s nearest neighbours become its slot-mates, not its synonyms | `shopper`, `client`, `buyer`, `consumer`, `subscriber` |

Claim 7 is the conceptual one: these vectors encode **interchangeability in context**, not
meaning. The model should learn that `customer` and `shopper` are similar because they fill
the same blank — never because it knows what a customer is.

Generated text should go from word salad at step 0 to locally grammatical but semantically
repetitive template sentences. Temperature should trade repetition for variety, and **no
weights should change during generation**.

*Note on method: the upstream repo ships `examples/reference/history.json`, a reference run
at this exact configuration. It was deliberately not opened before these numbers were
committed, so these are predictions rather than lookups.*

---

## 5. The corpus

### Sources and permissions

⬜ **PENDING**

All extension material in [`corpus/extensions/`](corpus/extensions/) is synthetic text written
specifically for this assignment. No third-party, confidential, or personal material is used,
so it is safe to publish. (Upstream git-ignores all of `corpus/`; this repo deliberately
un-ignores `corpus/extensions/` so the teaching examples are readable — see
[`.gitignore`](.gitignore).)

No PDFs are used, so there is no PDF extraction to verify and no extraction warnings to
resolve. If that changes, the check is to read the extracted text back out of the saved
`corpus.txt` and the previews in `corpus_manifest.json` rather than trusting the file count.

### Measured corpus facts

| | Experiment 1 (starter) | Experiment 2 (extended) |
|---|---|---|
| Corpus mode | ⬜ PENDING | ⬜ PENDING |
| Unique passages | ⬜ PENDING | ⬜ PENDING |
| Vocabulary size | ⬜ PENDING | ⬜ PENDING |
| Training unknown-token rate | ⬜ PENDING | ⬜ PENDING |
| Held-out unknown-token rate | ⬜ PENDING | ⬜ PENDING |
| Train / validation split | ⬜ PENDING | ⬜ PENDING |
| Reserved eval passages | ⬜ PENDING | ⬜ PENDING |

Links: `corpus_manifest.json` · `vocabulary_report.json` (per experiment, under `evidence/`).

The tokenizer keeps only the **509 most frequent training token types**; everything else
becomes `<UNK>`. The split is by short passage, not by source file — which limits what the
held-out set can actually test, since two passages from the same template generator can land
on opposite sides of the split and look nearly identical.

---

## 6. What actually happened

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Completed steps | ⬜ PENDING | ⬜ PENDING |
| Elapsed time | ⬜ PENDING | ⬜ PENDING |
| Hardware | ⬜ PENDING | ⬜ PENDING |
| Parameters | ⬜ PENDING | ⬜ PENDING |
| Interrupted / failed? | ⬜ PENDING | ⬜ PENDING |

Links: executed notebooks in [`notebooks/`](notebooks/), plus `config.json`,
`training.csv`, `training_summary.json` per experiment.

**What I expected vs. what I observed:** ⬜ PENDING

---

## 7. Evidence

### Loss curves

⬜ **PENDING** — `training_curves.svg` embedded here, plus the complete loss table from
`history.json`.

These are **fixed evaluation panels**: at most 20 training documents and 20 validation
documents, the same ones at every measurement, reduced as the mean over non-padding
next-token targets. They are not a full-corpus loss.

### Text samples: untrained, halfway, final

⬜ **PENDING** — all three at identical generation settings, with the full files linked and
at least one visible change (or lack of change) explained. Empty or garbled output gets shown
as-is.

### Token → ID → vector, and one real weight update

⬜ **PENDING** — `tokenization.json` and `inspection.json`. The notebook probes the word
`customer` (token ID 28 in the starter vocabulary) and saves its 64-number vector before and
after training, the first parameter's value / gradient / post-update value, and a next-token
probability comparison for the same prefix.

### Temperature comparison

⬜ **PENDING** — three temperatures, `temperature_comparison.json`. Temperature changes only
how the next token is *sampled* from the probability distribution; **no weights change during
generation.**

---

## 8. The 48-case language evals

The suite lives at [`evals/language_evals.json`](evals/language_evals.json) and is byte-identical
to upstream (hash above). The runner is [`run_evals.py`](run_evals.py), which performs inference
and scoring only — it never trains.

### How scoring works

1. Only the **prefix** is fed to the model. Answers and answer choices are never in the prompt.
2. The four candidate words' next-token probabilities are compared; highest wins. Correct = 1,
   incorrect or tied = 0. Random guessing averages 25%.
3. Separately, an **unconstrained continuation** is generated at temperature 0.8 with a fixed
   per-case seed and a 24-token cap. This free text is saved but **is not what the
   multiple-choice score grades.**
4. If any word in the prompt or choices is outside the learned vocabulary, the case is marked
   `out_of_vocabulary` and scores zero — no lucky credit for comparing identical `<UNK>` tokens.

This is why three different numbers get reported, and they answer different questions:

- **All-case success (/48)** — the honest headline. Missing coverage counts as zero, so you
  cannot inflate it by dropping hard cases.
- **Accuracy among scorable cases** — how well the model does on questions it *could* answer.
  This can move simply because coverage changed, so it must be read next to coverage.
- **Vocabulary coverage** — what fraction of cases the model had the words for at all. This is
  a property of the corpus, not of the model's reasoning.

### Required four-row comparison

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable | Coverage | Full results |
|---|---|---|---|---|---|---|
| Starter corpus | Untrained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Starter corpus | Trained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Extended corpus | Untrained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Extended corpus | Trained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### Breakdown by group and category

⬜ **PENDING** — all 48 cases, per-group and per-category, both experiments.

### Actual free continuations

⬜ **PENDING** — shown separately from the multiple-choice score, including empty and
nonsensical ones.

### Keeping the exam out of the textbook

⬜ **PENDING** — `eval_separation.json` per experiment.

The notebook reserves any generated classroom passage containing a test prefix *before* the
train/validation split and before vocabulary building, rejects exact test prefixes in imported
files, and refuses a corpus folder that contains `evals/` or the repo root.

**Limits of this check, stated plainly:** it normalizes case, punctuation spacing, and
whitespace, and catches *exact* prefix matches. It does **not** detect paraphrases, leaked
answer lists, or semantic contamination. Separation therefore depends on the extension
material being written independently, not just on the automated check passing.

**These are public development tests, not an unseen final benchmark.** They were visible while
the extension corpus was being written, so results here measure guided development, not
generalization to unseen data.

---

## 9. Corpus extension experiment

**Categories chosen (at least two required):** ⬜ PENDING

**Why these categories, and what gap the new material addresses:** ⬜ PENDING

**The teaching material:** [`corpus/extensions/`](corpus/extensions/) ⬜ PENDING

Written as varied practice sentences, not one copied answer per test case. The eval prompts,
answer choices, answer key, scoring rules, eval outputs, and chat logs are never used as
training text or as a vocabulary source.

> **Gotcha worth recording.** `custom_llm.py:175` skips exactly one file — the top-level
> `corpus/README.md`. Every other `.md`, `.txt`, and `.pdf` under `corpus/`, *including in
> subfolders*, is ingested as training text. So there is deliberately **no README inside
> `corpus/extensions/`**: adding one would silently train the model on prose about the
> assignment. Documentation about the extension material lives in
> [`docs/`](docs/) and in this README instead, outside the corpus tree.

**Starter vs. extended comparison:** ⬜ PENDING — including whether any change reflects
vocabulary coverage, learned patterns, or both. Failures are reported, not hidden.

---

## 10. Chat interface

**Launch instructions:** ⬜ PENDING

Terminal:

```bash
./.venv/bin/python chat.py --model llm_runs/YOUR_RUN/model.pt --transcript results/my-chat.json
```

Type `/quit` to exit. Or use Section 10 of the notebook: edit `CHAT_PROMPT`, run the cell.

**Model / run identity:** ⬜ PENDING (run folder + model SHA-256)

**At least three real interactions:** ⬜ PENDING — from `chat_transcript.json`, actual prompts
and actual replies from the trained nanoGPT. Not canned text, not another model.

**One observed limitation:** ⬜ PENDING — covering unknown words, the 48-token context limit,
and the fact that each message starts fresh with **no shared history between prompts**.

---

## 11. What I learned — in my own words

> These are mine to write, not the assistant's. Each one is answered against actual values
> from my run.

1. **What is my corpus, what can it teach, and what is missing? Why hold data out?** ⬜ PENDING
2. **How do a token, a token ID, a vector, and an embedding differ?** ⬜ PENDING
3. **What makes this a neural network? How did loss, gradients, and the optimizer change its
   weights?** ⬜ PENDING
4. **What does attention combine, and why can it not look at future tokens?** ⬜ PENDING
5. **How do probabilities become generated text? What changed with temperature, and did any
   weights change then?** ⬜ PENDING
6. **Did the samples and both loss curves support my prediction? What can I honestly
   conclude?** ⬜ PENDING

---

## 12. One limitation and my next experiment

**Observed limitation:** ⬜ PENDING

**Proposed next experiment, and predicted effect:** ⬜ PENDING

---

## 13. Reproduce and inspect

| What | Where |
|---|---|
| Executed notebooks | [`notebooks/`](notebooks/) |
| Eval suite (unchanged) | [`evals/language_evals.json`](evals/language_evals.json) |
| Eval runner (inference only) | [`run_evals.py`](run_evals.py) |
| All eval results | [`evidence/`](evidence/) |
| Extension teaching material | [`corpus/extensions/`](corpus/extensions/) |
| Chat interface | [`chat.py`](chat.py) |
| Embedding viewer | [`embedding-viewer.html`](embedding-viewer.html) — open locally, load a run's `checkpoint.json` |
| nanoGPT model source | [`nanogpt_model.py`](nanogpt_model.py) (pinned + hash-checked) |
| Assignment brief | [`docs/assignment-brief.txt`](docs/assignment-brief.txt) |

Re-run the evals against a saved model without retraining:

```bash
./.venv/bin/python run_evals.py --model llm_runs/YOUR_RUN/model.pt --output results/final-evals
./.venv/bin/python run_evals.py --model llm_runs/YOUR_RUN/model_untrained.pt --stage untrained --output results/untrained-evals
```

Maintainer checks:

```bash
./.venv/bin/python -m unittest test_language_evals test_corpus
```

`checkpoint.json` stores initial and final embeddings for the viewer; `model.pt` stores the
full network for inference. Neither is an exact training-resume file.

---

## Honesty statement

No result in this README is invented, estimated, or carried over from the reference run in
the upstream repository. The eval suite is unchanged and hash-verified. Eval prompts, answer
keys, and eval outputs are not used as training data or as vocabulary sources. Every model
reply shown comes from the nanoGPT trained in this repository. Where the model fails, the
failure is shown.
