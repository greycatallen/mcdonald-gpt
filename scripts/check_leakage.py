"""Pre-flight check: does any corpus file contain an eval prompt?

The notebook already refuses to train on a leaking corpus, but it aborts on the
FIRST offending file, so a single run does not tell you whether the rest are clean.
This audits every file against all 48 cases using the notebook's own
`matching_cases()`, so the normalization rules are identical.

Run:  python scripts/check_leakage.py corpus_sets/targeted
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_evals import load_suite, matching_cases  # noqa: E402

SUITE = load_suite("evals/language_evals.json")


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus_sets/targeted")
    files = sorted(p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"})
    if not files:
        print(f"no .txt/.md/.pdf files under {folder}")
        return 0

    total = 0
    for path in files:
        if path.suffix.lower() == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(path)
            if reader.is_encrypted:
                reader.decrypt("")
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            text = path.read_text(encoding="utf-8")
        hits = matching_cases(text, SUITE)
        total += len(hits)
        if hits:
            print(f"LEAK  {path.name}: {', '.join(hits)}")
            for case in SUITE["cases"]:
                if case["id"] in hits:
                    for line in text.splitlines():
                        if matching_cases(line, SUITE) and case["id"] in matching_cases(line, SUITE):
                            print(f"         {case['id']} prompt : {case['prompt']!r}")
                            print(f"         offending line : {line!r}")
                            break
        else:
            print(f"clean {path.name}")

    print()
    print("RESULT:", "CLEAN - no eval prompt appears in any file" if total == 0
          else f"{total} leak(s) found - fix before training")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
