import unittest
from sympy import symbols
from ParallelGame import TwoLinkParallelGame


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
        game = TwoLinkParallelGame([[1, 0], [a, 0]])
        br1 = [
            {'cond': game.t2 < a + 2, 'func': (game.t2 + a) / 2},
            {'cond': game.t2 >= a + 2, 'func': game.t2 - 1}
        ]
        br2 = [
            {'cond': game.t1 < 2 * a + 1, 'func': (game.t1 + 1) / 2},
            {'cond': game.t1 >= 2 * a + 1, 'func': game.t1 - a}
        ]

        game.calculate_best_responses()
        self.assertEqual(game.br1, br1)
        self.assertEqual(game.br2, br2)
