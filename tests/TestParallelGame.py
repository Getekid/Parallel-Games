import unittest
from src import ParallelGame
import numpy as np


class TestParallelGame(unittest.TestCase):
    @staticmethod
    def test_get_flow():
        game = ParallelGame([[2, 0], [1, 1], [1, 2]])
        np.testing.assert_array_equal(game.get_flow(), [2/3, 1/3, 0])
        np.testing.assert_array_equal(game.get_flow([4, 2, 1]), [0, 1/2, 1/2])
        game = ParallelGame([[2, 0], [1, 1], [1, 0]])
        np.testing.assert_array_equal(game.get_flow(), [1/3, 0, 2/3])
        game = ParallelGame([[2, 0], [1, 1], [1, 1]])
        np.testing.assert_array_equal(game.get_flow(), [3/5, 1/5, 1/5])
        np.testing.assert_array_equal(game.get_flow([3/4, 2/4, 1/4]), [0.45, 0.15, 0.4 ])
