import unittest

from experiments.unknown_mass import estimate_unseen_fraction


class UnknownMassTests(unittest.TestCase):
  def test_duplicates_are_removed_before_capture_recapture(self):
    estimate = estimate_unseen_fraction(["a", "a", "b"], ["b", "b", "c"])
    self.assertEqual((estimate.first_count, estimate.second_count), (2, 2))
    self.assertEqual((estimate.overlap_count, estimate.union_count), (1, 3))
    self.assertGreater(estimate.unseen_fraction, 0.0)


  def test_full_overlap_has_no_unseen_class_reserve(self):
    estimate = estimate_unseen_fraction(["a", "b"], ["b", "a"])
    self.assertEqual(estimate.unseen_fraction, 0.0)
    self.assertTrue(estimate.reliable)


  def test_disjoint_panels_are_maximally_unknown_and_unreliable(self):
    estimate = estimate_unseen_fraction(["a"], ["b"])
    self.assertEqual(estimate.unseen_fraction, 1.0)
    self.assertFalse(estimate.reliable)


  def test_nested_grids_are_canonicalized(self):
    estimate = estimate_unseen_fraction(
        [[ [0, 1], [1, 0] ], [[0, 1], [1, 0]]],
        [[[0, 1], [1, 0]]],
    )
    self.assertEqual(estimate.overlap_count, 1)
