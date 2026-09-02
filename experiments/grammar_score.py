"""Conditional log-likelihood for a sound constrained grid grammar."""

from __future__ import annotations

from math import exp, isfinite, log
from typing import Iterable, Mapping


def logsumexp(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(not isfinite(value) for value in values):
        raise ValueError("values must be a non-empty finite sequence")
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def full_vocab_nll(logits: Mapping[int, float], target: int) -> float:
    """NLL using the original model's full-vocabulary normalization."""

    if not logits or target not in logits:
        raise ValueError("target must occur in a non-empty logits mapping")
    return logsumexp(logits.values()) - float(logits[target])


def conditional_grammar_nll(
    logits: Mapping[int, float],
    allowed_tokens: Iterable[int],
    target: int,
) -> float:
    """NLL after conditioning the model on the grammar's legal token set."""

    allowed = tuple(dict.fromkeys(int(token) for token in allowed_tokens))
    if not allowed or any(token not in logits for token in allowed):
        raise ValueError("allowed tokens must be a non-empty subset of logits")
    if target not in allowed:
        raise ValueError("target token is illegal under the grammar")
    return logsumexp(logits[token] for token in allowed) - float(logits[target])


def invalid_mass_nll_gain(
    logits: Mapping[int, float],
    allowed_tokens: Iterable[int],
) -> float:
    """NLL removed by conditioning on legal tokens at one state."""

    allowed = tuple(dict.fromkeys(int(token) for token in allowed_tokens))
    if not allowed or any(token not in logits for token in allowed):
        raise ValueError("allowed tokens must be a non-empty subset of logits")
    return logsumexp(logits.values()) - logsumexp(logits[token] for token in allowed)


if __name__ == "__main__":
    logits = {0: 0.0, 1: 0.0, 2: 0.0}
    assert conditional_grammar_nll(logits, (0,), 0) == 0.0
    assert invalid_mass_nll_gain(logits, (0, 1)) > 0.0
    assert full_vocab_nll(logits, 0) > conditional_grammar_nll(logits, (0, 1), 0)
    print("grammar_score selftest: PASS")
