"""Audit the ACTUAL training text of every experiment for eval contamination.

The notebook's own guard checks imported files at load time. This checks the corpus
that each model was really trained on (`evidence/*/corpus.txt`), and goes past exact
prompt matching to the things exact matching misses.

    1. eval prompt verbatim in the training text
    2. prompt + its answer, together
    3. any line containing all four answer choices of a case (a leaked answer list)
    4. eval/chat/results files sitting inside a corpus folder
    5. the reservation actually withheld passages before the split
    6. the answer co-occurring with its own prompt context on one line

Check 6 is the important one: contamination that matters is the model having seen the
question next to its answer. None of this detects paraphrase or semantic overlap --
see the disclosure in README section 8.

Run:  python scripts/audit_leakage.py
Exit: 0 clean, 1 if anything is flagged.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_evals import load_suite, matching_cases, normalized  # noqa: E402

suite = load_suite("evals/language_evals.json")
cases = suite["cases"]
EXPS = [("experiment1-starter", "Exp 1 starter"),
        ("experiment2-mcdonalds", "Exp 2 McDonald's"),
        ("experiment3-targeted", "Exp 3 targeted")]
FAILED = []


def corpus(d):
    return Path(f"evidence/{d}/corpus.txt").read_text(errors="replace")


def check(title, fn):
    print("=" * 78); print(title); print("=" * 78)
    for d, label in EXPS:
        msg, bad = fn(d)
        if bad:
            FAILED.append((label, title))
        print(f"  {label:20s} {msg}")
    print()


check("CHECK 1 - eval PROMPT verbatim in the real training text",
      lambda d: ((f"LEAK: {matching_cases(corpus(d), suite)}" if matching_cases(corpus(d), suite)
                  else "clean (0 of 48 prompts present)"), bool(matching_cases(corpus(d), suite))))


def c2(d):
    n = normalized(corpus(d))
    hits = [c["id"] for c in cases if normalized(c["prompt"] + " " + c["answer"]) in n]
    return (f"LEAK: {hits}" if hits else "clean (0 of 48 prompt+answer pairs present)"), bool(hits)


check("CHECK 2 - prompt + its ANSWER together in the training text", c2)


def c3(d):
    bad = []
    for line in corpus(d).split("\n"):
        nl = normalized(line)
        if not nl:
            continue
        for c in cases:
            if sum(1 for ch in c["choices"] if f" {normalized(ch)} " in f" {nl} ") == 4:
                bad.append(c["id"]); break
    return (f"LEAK: {bad[:5]}" if bad else "clean (no line holds all 4 choices of any case)"), bool(bad)


check("CHECK 3 - leaked ANSWER LIST (all four choices on one line)", c3)


def c6(d):
    flags = []
    lines = [normalized(l) for l in corpus(d).split("\n") if l.strip()]
    for c in cases:
        pw = set(normalized(c["prompt"]).split()) - {"the", "a", "is", "of", "to", "in", "and", ".", "it"}
        ans = normalized(c["answer"])
        if not pw:
            continue
        for nl in lines:
            ws = set(nl.split())
            if ans in ws and len(pw & ws) / len(pw) >= 0.6:
                flags.append(c["id"]); break
    return (f"LEAK: {flags[:5]}" if flags else "clean (answer never co-occurs with its prompt context)"), bool(flags)


check("CHECK 6 - ANSWER adjacent to its own PROMPT CONTEXT on one line", c6)

print("=" * 78); print("CHECK 4 - eval/chat/results files inside a corpus folder"); print("=" * 78)
stray = [str(p) for p in Path("corpus_sets").rglob("*")
         if p.is_file() and any(k in p.name.lower()
                                for k in ("eval", "chat", "transcript", "summary", "results"))]
stray += [str(p) for p in Path("corpus").rglob("*") if p.is_file() and p.name != "README.md"]
if stray:
    FAILED.append(("all", "stray files"))
print("  " + (f"LEAK: {stray}" if stray else "clean - no eval/chat/results file in any corpus folder"))
print()

print("=" * 78); print("CHECK 5 - reservation actually withheld passages before the split"); print("=" * 78)
for d, label in EXPS:
    s = json.load(open(f"evidence/{d}/eval_separation.json"))
    if s["excluded_passages"] == 0:
        FAILED.append((label, "no passages reserved"))
    print(f"  {label:20s} excluded={s['excluded_passages']:4d}  cases_protected={len(s['case_ids'])}")
print()

print("=" * 78)
print("RESULT:", "ALL CHECKS CLEAN" if not FAILED else f"{len(FAILED)} FAILURE(S): {FAILED}")
print("=" * 78)
print("Scope: exact/near-exact matching only. Does NOT detect paraphrase or semantic")
print("overlap. See README section 8 for the disclosed overlap in taught object pairs.")
sys.exit(1 if FAILED else 0)
