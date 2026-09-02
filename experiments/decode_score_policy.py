"""Length-aware score policies for ARC decoder frontier experiments.

This module is a CPU-only contract for replaying a decoder cache.  It does not
change the Kaggle notebook or generate candidates.  The absolute policy
matches the current cumulative-NLL cutoff; normalized and MDL policies expose
the two research alternatives documented in the autoresearch log.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Literal


PolicyMode = Literal["absolute", "mean_nll", "mdl"]


@dataclass(frozen=True)
class DecodeScorePolicy:
    """A complete-sequence acceptance policy for cache/frontier replay."""

    mode: PolicyMode
    budget: float
    length_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.budget <= 0.0:
            raise ValueError("budget must be positive")
        if self.length_penalty < 0.0:
            raise ValueError("length_penalty must be non-negative")
        if self.mode != "mdl" and self.length_penalty != 0.0:
            raise ValueError("length_penalty is only valid for mdl mode")

    def score(self, nll: float, length: int) -> float:
        """Return the policy score, with lower values preferred."""

        if nll < 0.0 or length < 1:
            raise ValueError("nll must be non-negative and length positive")
        if self.mode == "absolute":
            return nll
        if self.mode == "mean_nll":
            return nll / length
        return nll + self.length_penalty * length

    def accepts(self, nll: float, length: int) -> bool:
        """Return whether a complete candidate lies inside the frontier."""

        return self.score(nll, length) < self.budget

    def absolute_mean_likelihood_requirement(self, length: int) -> float:
        """Mean token likelihood required by an absolute-NLL policy."""

        if length < 1:
            raise ValueError("length must be positive")
        if self.mode != "absolute":
            raise ValueError("only absolute policy has this requirement")
        return exp(-self.budget / length)

    def optimistic_partial_score(
        self, partial_nll: float, emitted: int, target_length: int
    ) -> float:
        """Lower-bound a partial path's eventual score optimistically.

        Remaining token NLL is optimistically treated as zero.  A caller may
        use this value only as a branch-admission bound; grammar validity,
        shape, palette, and EOS checks remain separate obligations.
        """

        if partial_nll < 0.0 or emitted < 0 or target_length < 1:
            raise ValueError("invalid partial path")
        if emitted > target_length:
            raise ValueError("emitted cannot exceed target_length")
        if self.mode == "absolute":
            return partial_nll
        if self.mode == "mean_nll":
            return partial_nll / target_length
        return partial_nll + self.length_penalty * target_length


def current_absolute_policy() -> DecodeScorePolicy:
    """Return the current notebook-equivalent cumulative-NLL policy."""

    return DecodeScorePolicy("absolute", -log(0.2))


if __name__ == "__main__":
    policy = current_absolute_policy()
    assert policy.absolute_mean_likelihood_requirement(100) < 0.985
    assert policy.absolute_mean_likelihood_requirement(900) > 0.998
    assert DecodeScorePolicy("mean_nll", 0.02).accepts(1.0, 100)
    assert DecodeScorePolicy("mdl", 4.0, 0.01).accepts(2.0, 100)
    print("decode_score_policy selftest: PASS")
