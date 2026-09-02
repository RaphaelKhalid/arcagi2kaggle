from __future__ import annotations

import pytest

from experiments.grid_prefix_grammar import EOS_TOKEN, NEWLINE_TOKEN, grammar_for


def test_legal_prefix_reaches_exact_terminal_state() -> None:
    grammar = grammar_for(2, 2, (0, 1))
    state = grammar.initial
    for token in (0, 1, NEWLINE_TOKEN, 1, 0, EOS_TOKEN):
        state = grammar.advance(state, token)
    assert grammar.is_complete(state)
    assert grammar.exact_token_count == 6


def test_palette_branching_and_completion_count_are_exact() -> None:
    grammar = grammar_for(2, 2, (0, 1))
    assert grammar.allowed_tokens(grammar.initial) == (0, 1)
    assert grammar.completion_count(grammar.initial) == 16
    after_cell = grammar.advance(grammar.initial, 0)
    assert grammar.completion_count(after_cell) == 8


def test_structure_tokens_are_forced() -> None:
    grammar = grammar_for(1, 1, (3,))
    after_cell = grammar.advance(grammar.initial, 3)
    assert grammar.allowed_tokens(after_cell) == (EOS_TOKEN,)
    with pytest.raises(ValueError):
        grammar.advance(after_cell, NEWLINE_TOKEN)


def test_invalid_dimensions_palette_and_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        grammar_for(0, 2, (0,))
    with pytest.raises(ValueError):
        grammar_for(2, 2, (10,))
    grammar = grammar_for(2, 2, (0,))
    with pytest.raises(ValueError):
        grammar.advance(grammar.initial, EOS_TOKEN)

