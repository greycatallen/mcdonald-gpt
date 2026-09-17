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
> | Prediction written and locked | ✅ committed `16faf8e`, before any graded run |
> | Experiment 1: starter corpus | ✅ done — [`evidence/experiment1-starter/`](evidence/experiment1-starter/) |
> | Experiment 2: + McDonald's filings | ⬜ pending |
> | Experiment 3: + category-targeted material | ⬜ pending |
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

### Scorecard: Experiment 1 predictions vs. measured results

**Three of seven claims were wrong.** They are reported here first, before the results
that were right.

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| 1 | Validation loss | 1.6 – 2.2 | **0.706** | ❌ **Wrong** — loss fell far further than expected |
| 2 | Train/val gap (no overfitting) | < 0.2 | **0.028** | ✅ Correct |
| 3 | `extend_corpus` | 0/24, 0% coverage | **0/24, 0% coverage** | ✅ Correct |
| 4 | `starter_patterns` | 60 – 85% | **100%** (16/16) | ❌ **Wrong** — exceeded the range |
| 5 | `starter_transfer` | 40 – 60% | **50%** (4/8) | ✅ Correct |
| 6 | Overall | 14 – 19 / 48 | **20 / 48** | ❌ **Wrong** — by one case |
| 7 | `customer`'s neighbours | `shopper`, `client`, `buyer`, `consumer`, `subscriber` | **exactly those five** | ✅ Correct |

**What I got wrong and why.** Claims 1, 4 and 6 all failed in the same direction: I badly
underestimated how learnable the starter corpus is. It is generated from a small set of
sentence templates, so once the model has the templates there is very little residual
uncertainty — validation loss reached 0.706 rather than the predicted 1.6–2.2, and
`starter_patterns` saturated at a perfect 16/16 instead of the predicted 60–85%. Claim 6
then failed as an arithmetic consequence of claim 4 being wrong, missing by a single case.
The lesson is that I was reasoning about "a tiny model on a small corpus" when the binding
constraint was actually **how repetitive the corpus is**, not how small the model is.

Claim 2 was the deliberately contrarian one, and it held: the gap was 0.028, not the
textbook overfitting divergence. Claim 7 held precisely, which is covered in §7.

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

| | Exp 1 (starter) | Exp 2 (+ McDonald's) | Exp 3 (+ targeted) |
|---|---|---|---|
| Corpus mode | `classroom`, 0 files | ⬜ PENDING | ⬜ PENDING |
| Corpus folder | `corpus_sets/starter` (empty) | `corpus_sets/mcdonalds` | `corpus_sets/targeted` |
| Vocabulary size | **136** | ⬜ PENDING | ⬜ PENDING |
| Training unknown-token rate | **0.0%** | ⬜ PENDING | ⬜ PENDING |
| Held-out unknown-token rate | **0.0%** | ⬜ PENDING | ⬜ PENDING |
| Train / validation split | **4132 / 460** | ⬜ PENDING | ⬜ PENDING |
| Reserved eval passages | **160** | ⬜ PENDING | ⬜ PENDING |

Experiment 1's vocabulary is only 136 types — well under the 509 cap — because the starter
corpus simply does not contain more distinct words than that. **The 509 cap does not bind
in Experiment 1, but it is expected to bind hard in Experiment 2**, which is the mechanism
behind the counter-prediction in §4.

**Why each experiment has its own corpus folder.** `CORPUS = "classroom"` means *classroom
sentences **plus** every ingestible file in `CORPUS_FOLDER`.* The McDonald's PDFs were
initially placed in the default `corpus/` folder, which would have silently added ~46 MB of
financial prose to the "starter-only" baseline and invalidated the comparison. Each run now
points at an isolated folder under `corpus_sets/`, and the runner prints the ingestible file
count at the top of every execution so contamination is visible rather than assumed.
Experiment 1's log reads `corpus folder : corpus_sets/starter (0 ingestible file(s))`.

Links: `corpus_manifest.json` · `vocabulary_report.json` (per experiment, under `evidence/`).

The tokenizer keeps only the **509 most frequent training token types**; everything else
becomes `<UNK>`. The split is by short passage, not by source file — which limits what the
held-out set can actually test, since two passages from the same template generator can land
on opposite sides of the split and look nearly identical.

---

## 6. What actually happened

| | Exp 1 (starter) | Exp 2 (+ McDonald's) | Exp 3 (+ targeted) |
|---|---|---|---|
| Completed steps | **3000 / 3000** | ⬜ PENDING | ⬜ PENDING |
| Training time | **13.4 s** | ⬜ PENDING | ⬜ PENDING |
| Full notebook wall clock | **21.8 s** | ⬜ PENDING | ⬜ PENDING |
| Hardware | macOS 26.6.2, arm64 (Apple Silicon), CPU | — | — |
| Parameters | **111,872** | ⬜ PENDING | ⬜ PENDING |
| Run folder | `20260917T204453_348548Z` | ⬜ PENDING | ⬜ PENDING |
| Interrupted / failed? | **No** | ⬜ PENDING | ⬜ PENDING |

Links: [`notebooks/experiment1-starter.executed.ipynb`](notebooks/experiment1-starter.executed.ipynb) ·
[`config.json`](evidence/experiment1-starter/config.json) ·
[`training.csv`](evidence/experiment1-starter/training.csv) ·
[`training_summary.json`](evidence/experiment1-starter/training_summary.json)

### Failures and interruptions, stated plainly

Nothing was interrupted, but three things went wrong during setup and are recorded rather
than quietly fixed:

1. **Evals crashed on a torch-only install.** `RuntimeError: Numpy is not available`, raised
   from `run_evals.py:80` where `model_hash()` calls `tensor.numpy()`. `numpy` is missing
   from `requirements.txt`. Fixed by installing it; documented in §2.
2. **An encrypted PDF stopped the whole import.** `McDonalds_2018_Annual_Report.pdf` raised
   `Could not import ...: encrypted PDF; export an unlocked copy you are allowed to use`.
   One bad file aborts the entire corpus load, not just that file. See §5.
3. **The starter baseline was nearly contaminated.** The McDonald's PDFs were sitting in the
   default `corpus/` folder while `CORPUS = "classroom"`, which would have folded them into
   the "starter-only" run. Caught only because failure (2) aborted the run first. Fixed
   structurally by giving each experiment its own corpus folder.

---

## 7. Evidence

### Loss curves — Experiment 1

![Training and validation loss, Experiment 1](evidence/experiment1-starter/training_curves.svg)

Complete measured values from [`history.json`](evidence/experiment1-starter/history.json):

| Step | Training panel loss | Validation panel loss | Gap |
|---:|---:|---:|---:|
| 0 | 4.9263 | 4.9275 | 0.0012 |
| 1500 | 0.6821 | 0.7182 | 0.0361 |
| 3000 | 0.6783 | 0.7061 | 0.0278 |

These are **fixed evaluation panels**: at most 20 training documents and 20 validation
documents, the same ones at every measurement, reduced as the mean over non-padding
next-token targets. They are not a full-corpus loss.

**Reading these numbers.** Step 0's loss of 4.9263 is not arbitrary — uniform guessing over
a 136-word vocabulary costs exactly ln(136) = **4.913**. The model starts as a random
guesser, by construction. By step 3000 it reaches 0.678/0.706.

**The validation gap never opens.** It is 0.036 at step 1500 and *narrows* to 0.028 by step
3000 — the validation curve tracks the training curve rather than turning upward. This
confirms prediction 2, and the reason matters: the split is by passage from the same template
generator, so held-out passages look almost exactly like training passages. **This is a weak
held-out set.** It demonstrates the model did not memorize specific passages, but it cannot
show generalization to genuinely new language — which is precisely what the `starter_transfer`
and `extend_corpus` eval groups are for, and where the model does much worse.

Almost all learning happens before step 1500: loss falls 4.93 → 0.68 in the first half, then
moves 0.0038 across the entire second half. Most of the 3000-step budget was spent going
nowhere.

### Text samples: untrained, halfway, final

Identical generation settings at all three points. Full files:
[`step_0000.txt`](evidence/experiment1-starter/samples/step_0000.txt) ·
[`step_1500.txt`](evidence/experiment1-starter/samples/step_1500.txt) ·
[`step_3000.txt`](evidence/experiment1-starter/samples/step_3000.txt)

**Step 0 (untrained):**
```
pear professor bond doctor course harvest team physician journey checking buyer delivery
traffic report the lecturer item offering and system <UNK> taste recommended mentioned bus
question customer at mortgage nurse in instructor
```

**Step 1500 (halfway):**
```
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
we learned about the important website during a discussion of data .
```

**Step 3000 (final):**
```
our school has a question about the new educator and lesson .
a review of risk helped us understand the different deposit .
the report about the nurse explains the health in detail .
the consumer compared the offering after checking the price .
```

**The visible change** is between step 0 and step 1500, and it is total: word salad with no
grammar, no sentence boundaries and a stray `<UNK>` becomes grammatical sentences with
correct article/noun/verb order and terminating periods.

**The equally important lack of change** is between 1500 and 3000: the first two lines are
*identical, word for word*. Only lines 3–4 differ. This is the sample-level counterpart of
the loss curve flattening — the second half of the training budget bought almost nothing.

### Token → ID → vector, and one real weight update

Files: [`tokenization.json`](evidence/experiment1-starter/tokenization.json) ·
[`inspection.json`](evidence/experiment1-starter/inspection.json)

The probe word is **`customer`**, token ID **28**.

**Its 64-number vector, first coordinate:** `-0.057592` before training → `+0.036634` after.
The word is not stored as text anywhere in the network; it is this row of 64 numbers.

**One real gradient and weight update** (the first optimizer step, coordinate 0):

| | Value |
|---|---|
| Weight before | `-0.057591915130615234` |
| Gradient | `+0.000692586530931294` |
| Learning rate at this step | `0.00001` (warmup — not the 0.001 peak) |
| Weight after | `-0.057601906359195710` |

The update is `-0.05759192 - (0.00001 × 0.00069) ≈ -0.05760191`. It moved the weight by about
**one hundred-thousandth**. A single step is essentially invisible; the visible change comes
from 3000 of them compounding. Note the learning rate here is 1e-05, not the configured
0.001, because the notebook applies **warmup** — the rate ramps up before cosine decay.

**Next-token probabilities for the prefix `the customer`:**

| Before training | | After training | |
|---|---:|---|---:|
| `customer` | 1.60% | `reviewed` | 17.82% |
| `bus` | 1.07% | `recommended` | 17.12% |
| `educator` | 1.04% | `ordered` | 16.85% |
| `us` | 1.03% | `selected` | 16.34% |
| `application` | 1.01% | `compared` | 15.97% |
| `course` | 0.98% | `returned` | 14.28% |

Before training the distribution is essentially flat — every word gets ~1/136 ≈ 0.74%, and
the top pick is the nonsensical `the customer customer`. After training, **98.4% of the
probability mass sits on six words, and all six are verbs a customer can perform.** The model
learned the *grammatical slot*, not the meaning of "customer".

**Neighbours of `customer` in embedding space** (cosine similarity over the full 64
dimensions):

| Before training | | After training | |
|---|---:|---|---:|
| `bus` | 0.213 | `shopper` | **0.978** |
| `educator` | 0.203 | `client` | **0.977** |
| `helped` | 0.202 | `buyer` | **0.977** |
| `bank` | 0.201 | `subscriber` | **0.971** |
| `risk` | 0.198 | `consumer` | **0.970** |
| `selected` | 0.191 | `team` | 0.503 |

Before training these are random initialization noise — `bus` and `helped` sit near
`customer` for no reason, all at ~0.2. After training the top five are exactly the five words
predicted in §4, all above 0.97, and then a **cliff to 0.503**. The model discovered that
`customer`, `shopper`, `client`, `buyer`, `subscriber` and `consumer` are interchangeable —
not because it knows what a customer is, but because they fill the same blanks. That
distinction is the single most important thing this run demonstrates.

### Temperature comparison

[`temperature_comparison.json`](evidence/experiment1-starter/temperature_comparison.json)

| Temperature | Fourth sampled line |
|---|---|
| 0.3 | `the local consumer was mentioned in the purchase report yesterday .` |
| 0.8 | `the consumer compared the offering after checking the price .` |
| 1.2 | `the consumer compared the offering after checking the price .` |

The first three lines are **identical at all three temperatures**; only the fourth differs,
and 0.8 and 1.2 produce the same text. This is a weaker temperature effect than expected,
and it follows directly from the probability table above: when six words split 98% of the
mass almost evenly (17.8% vs 14.3%), flattening or sharpening that distribution rarely
changes which one gets drawn. Temperature has little to work with when the model is this
confident and the corpus this templated.

**No weights changed during any of this.** Temperature is applied at sampling time, dividing
the logits before the softmax. The same trained network produced all three columns.

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
| Starter corpus | Untrained | **9** (18.75%) | 24 | 37.50% | 50% | [untrained/](evidence/experiment1-starter/language_evals/untrained/) |
| Starter corpus | Trained | **20** (41.67%) | 24 | **83.33%** | 50% | [final/](evidence/experiment1-starter/language_evals/final/) |
| + McDonald's | Untrained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| + McDonald's | Trained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| + targeted | Untrained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| + targeted | Trained | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

Training moved the starter model from 9 to 20 correct out of 48. Read that carefully: the
all-case rate went 18.75% → 41.67%, but the accuracy *among cases it could answer* went
37.50% → **83.33%**. The two numbers differ because **coverage stayed frozen at exactly 50%**
— training cannot add words to the vocabulary, so the 24 out-of-vocabulary cases were
unanswerable before and remain unanswerable after.

### Breakdown by group — Experiment 1

| Group | Untrained | Trained | Scorable | Coverage |
|---|---|---|---|---|
| `starter_patterns` | 6/16 (37.5%) | **16/16 (100%)** | 16 | 100% |
| `starter_transfer` | 3/8 (37.5%) | **4/8 (50%)** | 8 | 100% |
| `extend_corpus` | 0/24 | **0/24** | 0 | **0%** |

### Breakdown by category — Experiment 1

| Category | Untrained | Trained | Coverage |
|---|---|---|---|
| `domain_context` | 3/8 | **8/8** | 100% |
| `domain_place` | 3/8 | **8/8** | 100% |
| `new_wording` | 3/8 | **4/8** | 100% |
| `grammar` | 0/3 | 0/3 | 0% |
| `opposites` | 0/3 | 0/3 | 0% |
| `negation` | 0/3 | 0/3 | 0% |
| `reference` | 0/3 | 0/3 | 0% |
| `sequence` | 0/3 | 0/3 | 0% |
| `spatial_relations` | 0/3 | 0/3 | 0% |
| `everyday_knowledge` | 0/3 | 0/3 | 0% |
| `categories_and_analogies` | 0/3 | 0/3 | 0% |

**What worked:** the two pure association categories saturated at 8/8. These are exactly what
the starter corpus drills — `customer`→`service`, `surgeon`→`patient`.

**What did not:** `new_wording` reached only 4/8. **This is the most informative result in
the experiment.** Those cases use the *same vocabulary the model has mastered*, rearranged
into unfamiliar sentence shapes. The model scores 100% on familiar templates and 50% — coin
flip — the moment the phrasing changes. It learned templates, not the associations
underneath them. Every zero below `new_wording` is a 0% coverage row, not a reasoning
failure.

### Actual free continuations — different from the multiple-choice score

These are unconstrained generations at temperature 0.8, saved regardless of quality. The
multiple-choice score does **not** grade them.

| Case | Prompt | Untrained continuation | Trained continuation |
|---|---|---|---|
| `lang_01` | `the report about the customer explains the` | `pear professor item doctor course harvest team physician journey checking buyer delivery our report the lecturer item offering and system <UNK> taste recommended payment` | `service in detail .` |
| `lang_17` | `our hospital discussed the nurse and the` | `customer order bank taste` | `health at the hospital .` |
| `lang_25` | `one bird` | `deposit a office car new client we quality selected mango harvest about about student team hospital merchandise client helped harvest discussion professor after discussed` | `the new customer with another client at the store .` |

`lang_25` is the case worth studying. The model scores **0** on it — `bird`, `one` and all
four choices (`is`, `are`, `were`, `am`) are outside the 136-word vocabulary, so it is marked
`out_of_vocabulary`. Yet its free continuation is a **fluent, grammatical English sentence**.

That gap is the entire argument for reporting both numbers. Judging this model by its free
text would suggest it understands the prompt; it cannot even represent the words in it. The
fluent output is the model ignoring an unreadable prompt and emitting a generic
high-probability sentence. Coverage is what exposes this, and a free-text demo would hide it.

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
