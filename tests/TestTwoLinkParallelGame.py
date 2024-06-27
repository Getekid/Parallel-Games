import unittest
from sympy import symbols, Piecewise
from src import TwoLinkParallelGame, TwoLinkPricingGame


class TestTwoLinkParallelGame(unittest.TestCase):
    def test_calculate_equilibrium(self):
        a = symbols('a')
        game = TwoLinkParallelGame([[1, 0], [a, 0]])
        x1 = (game.t2 - game.t1 + a) / (a + 1)
        x2 = (game.t1 - game.t2 + 1) / (a + 1)
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)

    def test_calculate_best_responses(self):
        a = symbols('a')
        game = TwoLinkPricingGame([[1, 0], [a, 0]])
        br1 = Piecewise(((game.t2 + a) / 2, game.t2 < a + 2), (game.t2 - 1, game.t2 >= a + 2))
        br2 = Piecewise(((game.t1 + 1) / 2, game.t1 < 2 * a + 1), (game.t1 - a, game.t1 >= 2 * a + 1))

        game.calculate_best_responses()
        self.assertEqual(game.br1, br1)
        self.assertEqual(game.br2, br2)

    def test_approximate_pricing_equilibrium(self):
        game = TwoLinkPricingGame([[1, 0], [2, 0]])  # a = 2
        self.assertIsNone(game.approximate_pricing_equilibrium())

        game.calculate_best_responses()
        equilibrium = game.approximate_pricing_equilibrium()
        # Expected equilibrium at ((2 * a + 1) / 3, (a + 2) / 3) for a = 2
        self.assertAlmostEqual(5 / 3, equilibrium[0])
        self.assertAlmostEqual(4 / 3, equilibrium[1])

        equilibrium = game.approximate_pricing_equilibrium(max_iter=40, init=(1, 5))
        self.assertAlmostEqual(5 / 3, equilibrium[0])
        self.assertAlmostEqual(4 / 3, equilibrium[1])
