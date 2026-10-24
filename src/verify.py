"""Shared verifier for the S1 rebuild.

Used by BOTH the GRPO reward function and the RFT rejection sampler, so the
two methods see identical prompts, identical token cap and identical checking
- this is the control that makes the comparison meaningful (per arXiv:2610.25812).

Reward structure follows Bauer et al. (arXiv:2604.18381) counting rewards:
  - correctness: +1.0 exact match, 0.0 otherwise
  - format:      +0.1 'Answer: X', +0.05 accepted variant, -0.1 invalid
  total reward range ~[-0.1, 1.1] (we drop the paper's reasoning-step penalty
  which depends on a step-counting heuristic not fully specified in the text;
  documented deviation, see notes/deviations.md).
"""

from __future__ import annotations

import re

# Primary: 'Answer: <number>', tolerant of markdown bold (**Answer: 25**)
# and of trailing punctuation/asterisks after the number.
_RE_PRIMARY = re.compile(r"Answer:\s*\**\s*(-?\d+)", re.IGNORECASE)
# Accepted variants: LaTeX \boxed{...} or a bare final integer.
_RE_BOXED = re.compile(r"\\boxed\{(-?\d+)\}")
_RE_VARIANT = re.compile(r"(-?\d+)\s*\**\s*$")


def parse_answer(completion: str) -> tuple[int | None, float]:
    """Return (parsed_value, format_bonus). format_bonus in {0.1, 0.05, -0.1}."""
    matches = _RE_PRIMARY.findall(completion)
    if matches:
        return int(matches[-1]), 0.1  # last 'Answer:' wins
    m = _RE_BOXED.findall(completion)
    if m:
        return int(m[-1]), 0.05
    # variant: last line that is (essentially) a bare number
    for line in reversed([ln.strip() for ln in completion.strip().splitlines() if ln.strip()]):
        m2 = _RE_VARIANT.search(line)
        if m2 and m2.group(1) != "":
            return int(m2.group(1)), 0.05
    return None, -0.1


def reward(prompt_answer: int, completion: str) -> float:
    """Bauer-style counting reward in ~[-0.1, 1.1]."""
    value, format_bonus = parse_answer(completion)
    if value is None:
        return format_bonus  # -0.1
    correct = 1.0 if value == int(prompt_answer) else 0.0
    return correct + format_bonus


def is_correct(prompt_answer: int, completion: str) -> bool:
    """Strict pass/fail used for evaluation and RFT rejection sampling:
    parsed value must equal ground truth. Format bonus is ignored here so
    that RFT training set == actually-correct solutions."""
    value, _ = parse_answer(completion)
    return value is not None and value == int(prompt_answer)


if __name__ == "__main__":
    cases = [
        (16, "Let me count... multiples of 6 up to 100.\nAnswer: 16"),
        (16, "the answer is Answer: 17"),
        (16, "blah\n17"),
        (16, "no number here"),
    ]
    for ans, c in cases:
        print(repr(c[-20:]), "->", reward(ans, c), is_correct(ans, c))
