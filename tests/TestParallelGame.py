import unittest
from src import ParallelGame, LinDistParallelGame, StepDistParallelGame
import numpy as np


class TestParallelGame(unittest.TestCase):
    @staticmethod
    def test_get_flow():
        game = ParallelGame([[2, 0], [1, 1], [1, 2]])
        np.testing.assert_array_equal(game.get_flow(), [2 / 3, 1 / 3, 0])
        np.testing.assert_array_equal(game.get_flow([4, 2, 1]), [0, 1 / 2, 1 / 2])

        game = ParallelGame([[2, 0], [1, 1], [1, 0]])
        np.testing.assert_array_equal(game.get_flow(), [1 / 3, 0, 2 / 3])

        game = ParallelGame([[2, 0], [1, 1], [1, 1]])
        np.testing.assert_array_equal(game.get_flow(), [3 / 5, 1 / 5, 1 / 5])
        np.testing.assert_array_equal(game.get_flow([3 / 4, 2 / 4, 1 / 4]), [0.45, 0.15, 0.4])

    @staticmethod
    def test_get_pricing_equilibrium():
        game = ParallelGame([[2, 0], [1, 1], [1, 1]])
        np.testing.assert_array_equal(game.get_pricing_equilibrium(), [1, 1 / 2, 1 / 2])

    def test_get_pricing_equilibrium_samples(self):
        game = ParallelGame([[2, 0], [1, 1], [1, 2]])
        t_star = game.get_pricing_equilibrium()
        flow_star = game.get_flow(t_star)
        profits = flow_star * t_star

        for i in range(len(t_star)):
            t_var = t_star.copy()
            for _ in range(1000):
                t_var[i] = np.random.uniform(0, 2 * t_star[i])
                x_var = game.get_flow(t_var)
                self.assertLessEqual(x_var[i] * t_var[i], profits[i],
                                     "Found larger profit for operator {i} for toll {toll}.".format(i=i, toll=t_var[i]))

    @staticmethod
    def test_appr_pricing_equilibrium():
        game = ParallelGame([[2, 0], [1, 1], [1, 1]])
        np.testing.assert_array_almost_equal(game.appr_pricing_equilibrium()[-1], [1, 1 / 2, 1 / 2], 4)


class TestLinDistParallelGame(unittest.TestCase):
    @staticmethod
    def test_get_flow():
        game = LinDistParallelGame([[2, 0], [1, 1]], [1, 1])
        np.testing.assert_array_equal(game.get_flow(), [2 / 3, 1 / 3])
        np.testing.assert_array_equal(game.get_flow([1, 1]), [2 / 3, 1 / 3])
        np.testing.assert_array_equal(game.get_flow([3, 2]), [1 / 4, 3 / 4])

        game = LinDistParallelGame([[2, 0], [1, 1], [1, 1]], [1, 1])
        np.testing.assert_array_equal(game.get_flow(), [3 / 5, 1 / 5, 1 / 5])
        np.testing.assert_array_equal(game.get_flow([3, 3, 3]), [3 / 5, 1 / 5, 1 / 5])
        np.testing.assert_array_equal(game.get_flow([4, 3, 5 / 2]), [1 / 18, 3 / 18, 14 / 18])
        np.testing.assert_array_almost_equal(game.get_flow([4, 3, 3]), [1 / 7, 3 / 7, 3 / 7], 10)
        np.testing.assert_array_equal(game.get_flow([4, 3, 2]), [0, 0, 1])
        np.testing.assert_array_equal(game.get_flow([4, 3, 1]), [0, 0, 1])

    @staticmethod
    def test_appr_pricing_equilibrium():
        game = LinDistParallelGame([[1, 0], [2, 0]], [1, 1])
        np.testing.assert_array_almost_equal(game.appr_pricing_equilibrium()[-1], [1.1464, 0.9268], 4)

        game = LinDistParallelGame([[2, 0], [1, 1], [1, 1]], [0, 1])
        np.testing.assert_array_almost_equal(game.appr_pricing_equilibrium()[-1], [1, 1 / 2, 1 / 2], 4)

        # Next game can have 2 N.E. (0.77, 0.34, 0.33) and (0.77, 0.33, 0.34).
        game = LinDistParallelGame([[2, 0], [1, 1], [1, 1]], [1, 1], seed=10)
        np.testing.assert_array_almost_equal(game.appr_pricing_equilibrium()[-1],
                                             [0.77774878, 0.34034214, 0.33369089], 4)
        game = LinDistParallelGame([[2, 0], [1, 1], [1, 1]], [1, 1], seed=2)
        np.testing.assert_array_almost_equal(game.appr_pricing_equilibrium()[-1],
                                             [0.77774878, 0.33369089, 0.34034214], 4)


class TestStepDistParallelGame(unittest.TestCase):
    @staticmethod
    def test_get_flow():
        game = StepDistParallelGame([[2, 0], [1, 1]], [3, 4], [1 / 4])
        np.testing.assert_array_equal(game.get_flow(), [2 / 3, 1 / 3])
        np.testing.assert_array_equal(game.get_flow([1, 1]), [2 / 3, 1 / 3])
        np.testing.assert_array_equal(game.get_flow([3 / 4, 2 / 4]), [1 / 3, 2 / 3])

    def test_get_step_value(self):
        game = StepDistParallelGame([[2, 0], [1, 1]], [3, 4], [1 / 4])
        self.assertEqual(game.get_step_value(-1), 3)
        self.assertEqual(game.get_step_value(1 / 5), 3)
        self.assertEqual(game.get_step_value(2 / 5), 4)
        self.assertEqual(game.get_step_value(2), 4)

    def test_get_2_link_pricing_equilibrium(self):
        game = StepDistParallelGame([[2, 0], [1, 1]], [3, 4], [1 / 4])
        pr_eq = game.get_2_link_pricing_equilibrium()
        self.assertIsInstance(pr_eq, np.ndarray)
        np.testing.assert_array_almost_equal(pr_eq, [5 / 12, 4 / 12])

        game = StepDistParallelGame([[2, 0], [1, 1]], [2, 4], [1 / 4])
        self.assertFalse(game.get_2_link_pricing_equilibrium())
