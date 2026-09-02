from __future__ import annotations

import math

import pytest

from experiments.grammar_score import (
    conditional_grammar_nll,
    full_vocab_nll,
    invalid_mass_nll_gain,
)


def test_singleton_grammar_state_has_zero_conditional_nll() -> None:
    assert conditional_grammar_nll({0: -3.0, 1: 4.0}, (0,), 0) == pytest.approx(0.0)


def test_conditioning_removes_invalid_vocabulary_mass() -> None:
    logits = {0: 0.0, 1: 0.0, 2: 0.0}
    assert conditional_grammar_nll(logits, (0, 1), 0) < full_vocab_nll(logits, 0)
    assert invalid_mass_nll_gain(logits, (0, 1)) == pytest.approx(math.log(3 / 2))


def test_allowed_order_and_duplicates_do_not_change_score() -> None:
    logits = {0: 0.0, 1: 1.0, 2: -2.0}
    left = conditional_grammar_nll(logits, (0, 1), 1)
    right = conditional_grammar_nll(logits, (1, 0, 1), 1)
    assert left == pytest.approx(right)


def test_illegal_or_missing_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        conditional_grammar_nll({0: 0.0}, (0,), 1)
    with pytest.raises(ValueError):
        conditional_grammar_nll({0: 0.0}, (1,), 1)
    with pytest.raises(ValueError):
        full_vocab_nll({}, 0)
