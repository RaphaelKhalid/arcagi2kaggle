"""Exact prefix automaton for shape- and palette-constrained grid decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


NEWLINE_TOKEN = 10
EOS_TOKEN = 15


@dataclass(frozen=True)
class PrefixState:
    row: int
    column: int
    done: bool = False


@dataclass(frozen=True)
class GridPrefixGrammar:
    """Language of exactly ``height x width`` grids plus EOS.

    Cell tokens are integer color IDs.  A row separator is required between
    rows, and EOS is required immediately after the final cell.  This mirrors
    the 16-token grid vocabulary used by the baseline decoder.
    """

    height: int
    width: int
    palette: frozenset[int]

    def __post_init__(self) -> None:
        if not 1 <= self.height <= 30 or not 1 <= self.width <= 30:
            raise ValueError("grid dimensions must lie in [1, 30]")
        if not self.palette or not self.palette <= frozenset(range(10)):
            raise ValueError("palette must be a non-empty subset of colors 0..9")

    @property
    def initial(self) -> PrefixState:
        return PrefixState(0, 0)

    @property
    def exact_token_count(self) -> int:
        return self.height * self.width + self.height

    def _validate_state(self, state: PrefixState) -> None:
        if state.done:
            if (state.row, state.column) != (self.height, 0):
                raise ValueError("done state must be terminal")
            return
        if not 0 <= state.row < self.height or not 0 <= state.column <= self.width:
            raise ValueError("state is outside grammar bounds")

    def allowed_tokens(self, state: PrefixState) -> tuple[int, ...]:
        """Return exactly the tokens that can extend a legal prefix."""

        self._validate_state(state)
        if state.done:
            return ()
        if state.column < self.width:
            return tuple(sorted(self.palette))
        if state.row < self.height - 1:
            return (NEWLINE_TOKEN,)
        return (EOS_TOKEN,)

    def advance(self, state: PrefixState, token: int) -> PrefixState:
        """Consume one token or reject the prefix immediately."""

        self._validate_state(state)
        if token not in self.allowed_tokens(state):
            raise ValueError("token is illegal for this grid prefix")
        if state.column < self.width:
            return PrefixState(state.row, state.column + 1)
        if state.row < self.height - 1:
            return PrefixState(state.row + 1, 0)
        return PrefixState(self.height, 0, done=True)

    def is_complete(self, state: PrefixState) -> bool:
        self._validate_state(state)
        return state.done

    def completion_count(self, state: PrefixState) -> int:
        """Count legal token completions from a prefix, including EOS."""

        self._validate_state(state)
        if state.done:
            return 1
        remaining_cells = (self.width - state.column)
        remaining_cells += (self.height - state.row - 1) * self.width
        return len(self.palette) ** remaining_cells


def grammar_for(
    height: int,
    width: int,
    palette: Iterable[int],
) -> GridPrefixGrammar:
    """Construct a grammar while canonicalizing a palette iterable."""

    return GridPrefixGrammar(height, width, frozenset(int(color) for color in palette))


if __name__ == "__main__":
    grammar = grammar_for(2, 2, (0, 1))
    state = grammar.initial
    for token in (0, 1, NEWLINE_TOKEN, 1, 0, EOS_TOKEN):
        state = grammar.advance(state, token)
    assert grammar.is_complete(state)
    assert grammar.completion_count(grammar.initial) == 16
    print("grid_prefix_grammar selftest: PASS")
