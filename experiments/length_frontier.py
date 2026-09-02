"""Incumbent-preserving length-normalized DFS frontier policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnionLengthFrontier:
    """Union of the current absolute-NLL and a length-normalized frontier.

    The absolute branch is retained explicitly.  Consequently this policy can
    only add accepted complete paths relative to the current cutoff, provided
    the caller gives both branches the required compute budget.
    """

    absolute_budget: float
    mean_nll_budget: float
    target_length: int

    def __post_init__(self) -> None:
        if self.absolute_budget <= 0.0 or self.mean_nll_budget <= 0.0:
            raise ValueError("NLL budgets must be positive")
        if self.target_length < 1:
            raise ValueError("target_length must be positive")

    @property
    def union_budget(self) -> float:
        """Absolute NLL threshold equivalent to the two accepted branches."""

        return max(self.absolute_budget, self.mean_nll_budget * self.target_length)

    def accepts_complete(self, nll: float) -> bool:
        """Accept a target-length completion under either score branch."""

        if nll < 0.0:
            raise ValueError("nll must be non-negative")
        return nll < self.absolute_budget or nll / self.target_length < self.mean_nll_budget

    def accepts_partial(self, nll: float, emitted: int) -> bool:
        """Apply the optimistic remaining-NLL-zero admission bound."""

        if nll < 0.0 or emitted < 0 or emitted > self.target_length:
            raise ValueError("invalid partial path")
        # Remaining NLL is non-negative, so this is a necessary condition for
        # either complete-path branch. Grammar/EOS validity remains separate.
        return nll < self.union_budget

    def mean_nll(self, nll: float) -> float:
        """Return the target-length normalized score used by the new branch."""

        if nll < 0.0:
            raise ValueError("nll must be non-negative")
        return nll / self.target_length


if __name__ == "__main__":
    frontier = UnionLengthFrontier(1.609, 0.02, 900)
    assert frontier.union_budget == 18.0
    assert frontier.accepts_complete(10.0)
    assert frontier.accepts_complete(1.0)
    assert frontier.accepts_partial(17.9, 100)
    print("length_frontier selftest: PASS")
