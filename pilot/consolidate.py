"""Rule cards: merge several failure-evidence notes about similar tasks into ONE note that shows the pattern.

A failure note (coding.note_style: evidence) says "the checker expected X (int) but got Y (float)". One such note is easy
to ignore; five of them about similar functions, summarised as "expected int but got float, 5 of 5 times", state the rule.
A small model does not do that summarising reliably (it invented rules in our tests), so it is done here by plain code:
no model is involved, nothing can be hallucinated, and every line in the card is a real checker result.
"""
from __future__ import annotations

import re
from collections import Counter

MARK = "FAILED the project's check: "
CARD_HEAD = "Rule evidence from"
_CHECK = re.compile(r"-> expected (.+) \((\w+)\) but got (.+) \((\w+)\)$")


def is_evidence_note(text: str) -> bool:
    return MARK in text and "Outcome: INCORRECT" in text


def _pairs_from_note(text: str) -> list[tuple[str, str]]:
    """[(task_line, check_line)] from a single evidence note or from an existing rule card."""
    if text.startswith(CARD_HEAD):
        out, task = [], None
        for line in text.splitlines():
            if line.startswith("- Task: "):
                task = line[len("- Task: "):]
            elif line.startswith("  Check: ") and task is not None:
                out.append((task, line[len("  Check: "):]))
        return out
    task = next((l[len("Task: "):] for l in text.splitlines() if l.startswith("Task: ")), "")
    check = text.split(MARK, 1)[1].splitlines()[0].strip() if MARK in text else ""
    return [(task, check)]


def build_card(notes: list[str], max_lines: int = 6) -> str:
    """One rule card from evidence notes and/or earlier rule cards (newest evidence kept when over the limit)."""
    pairs: list[tuple[str, str]] = []
    for t in notes:
        pairs += _pairs_from_note(t)
    seen, uniq = set(), []
    for p in pairs:                                  # the same check twice adds nothing
        if p[1] not in seen:
            seen.add(p[1]); uniq.append(p)
    uniq = uniq[-max_lines:]
    lines = [f"{CARD_HEAD} {len(uniq)} earlier tasks of this kind (the project's checker rejected every attempt):"]
    for task, check in uniq:
        lines += [f"- Task: {task[:110]}", f"  Check: {check}"]
    kinds = Counter()
    for _, check in uniq:
        m = _CHECK.search(check)
        if m:
            kinds[(m.group(2), m.group(4))] += 1
    if kinds:
        (want, got), k = kinds.most_common(1)[0]
        n = sum(kinds.values())
        if want != got:
            lines.append(f"Pattern: in {k} of {n} failures the checker expected type {want} but the solution returned type {got}.")
        else:
            lines.append(f"Pattern: in {k} of {n} failures the type was right ({want}) but the value or format was not; "
                         f"match the expected values above exactly.")
    lines.append("Outcome: INCORRECT")
    return "\n".join(lines)


_SIG_LINE = re.compile(r"expected .+? \((\w+)\) but got .+? \((\w+)\)")
_EXC = re.compile(r"(?:FAILED the project's check|failed with|Check): .*?\b(\w+(?:Error|Exception))\b")
_NOT_A_CRASH = {"AssertionError"}      # a wrong answer, not a crash: its signature is the (expected type, got type) pair


def note_signatures(text: str) -> set:
    """The failure signatures a note is about: (expected type, got type) pairs and ('exception', name) pairs."""
    sigs = {(m.group(1), m.group(2)) for m in _SIG_LINE.finditer(text)}
    sigs |= {("exception", m.group(1)) for m in _EXC.finditer(text) if m.group(1) not in _NOT_A_CRASH}
    return sigs


def detail_signature(detail: str) -> set:
    """The signature of a fresh failure, from the checker's own message (before it is written into a note)."""
    m = re.search(r"expected (.+?), got (.+)$", detail)
    sigs = set()
    if m:
        import ast
        def kind(x):
            try:
                return type(ast.literal_eval(x)).__name__
            except Exception:
                return "?"
        sigs.add((kind(m.group(1)), kind(m.group(2))))
    e = re.match(r"(\w+(?:Error|Exception))\b", detail.strip())
    if e and e.group(1) not in _NOT_A_CRASH:
        sigs.add(("exception", e.group(1)))
    return sigs


def note_kind(text: str) -> str:
    """What sort of memory a note is: a rule card, a verified worked example, bare failure evidence, or a solved task."""
    if text.startswith(CARD_HEAD):
        return "card"
    if "Verified solution (" in text:
        return "example"
    if is_evidence_note(text):
        return "evidence"
    return "solved"
