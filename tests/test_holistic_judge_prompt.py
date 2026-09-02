from __future__ import annotations

import json
import unittest

from experiments.candidate_cards import deduplicate_cards
from experiments.candidate_records import CandidateRecord, grid_hash
from experiments.holistic_judge_prompt import (
    build_holistic_judge_prompt,
    parse_judge_choice,
)
from experiments.holistic_context import TraceEvidence


class HolisticJudgePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="a",
                output=[[0]],
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="b", candidate_id="b",
                output=[[1]],
            ),
        ]
        self.cards = deduplicate_cards(records)
        self.outputs = {grid_hash(((0,),)): [[0]], grid_hash(((1,),)): [[1]]}
        self.hashes = tuple(card.output_hash for card in self.cards)

    def test_bundle_is_deterministic_and_contains_exact_outputs(self):
        first = build_holistic_judge_prompt(self.cards, self.outputs)
        second = build_holistic_judge_prompt(tuple(reversed(self.cards)), self.outputs)
        self.assertEqual(first, second)
        for output_hash in self.hashes:
            self.assertIn(output_hash, first)
        self.assertIn("exact_output:\n0", first)

    def test_bundle_rejects_missing_or_oversized_content(self):
        with self.assertRaises(ValueError):
            build_holistic_judge_prompt(self.cards, {}, max_chars=1000)
        with self.assertRaises(ValueError):
            build_holistic_judge_prompt(self.cards, self.outputs, max_chars=10)
        wrong_outputs = dict(self.outputs)
        wrong_outputs[self.hashes[0]] = [[9]]
        with self.assertRaises(ValueError):
            build_holistic_judge_prompt(self.cards, wrong_outputs)

    def test_parser_accepts_only_two_known_distinct_classes(self):
        raw = "```json\n" + json.dumps({"top_two": list(self.hashes)}) + "\n```"
        self.assertEqual(parse_judge_choice(raw, self.hashes), self.hashes)
        with self.assertRaises(ValueError):
            parse_judge_choice(json.dumps({"top_two": [self.hashes[0], "invented"]}), self.hashes)
        with self.assertRaises(ValueError):
            parse_judge_choice(json.dumps({"top_two": [self.hashes[0], self.hashes[0]]}), self.hashes)

    def test_parser_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            parse_judge_choice("not json", self.hashes)
        with self.assertRaises(ValueError):
            parse_judge_choice(json.dumps({"top_two": [self.hashes[0]]}), self.hashes)

    def test_optional_traces_are_appended_after_exact_outputs(self):
        traces = [
            TraceEvidence("a-trace", self.hashes[0], "family-a", "proof a"),
            TraceEvidence("b-trace", self.hashes[1], "family-b", "proof b"),
        ]
        prompt = build_holistic_judge_prompt(
            self.cards,
            self.outputs,
            trace_evidence=traces,
        )
        self.assertLess(prompt.index("TRACE EVIDENCE"), len(prompt))
        self.assertIn("candidate=a-trace", prompt)
        self.assertIn("candidate=b-trace", prompt)

    def test_partial_trace_coverage_fails_closed(self):
        with self.assertRaises(ValueError):
            build_holistic_judge_prompt(
                self.cards,
                self.outputs,
                trace_evidence=[
                    TraceEvidence("a-trace", self.hashes[0], "family-a", "proof a")
                ],
            )


if __name__ == "__main__":
    unittest.main()
