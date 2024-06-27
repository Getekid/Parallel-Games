import unittest
from sympy import symbols, Piecewise, nan
from src import TwoLinkHeterogeneousGame


class TestTwoLinkHeterogeneousGame(unittest.TestCase):
    def test_calculate_equilibrium(self):
        a2, a, p = symbols('a2 a p', nonnegative=True)

        # Game with a = 1.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], a.subs(a, 1.0))
        x1 = (game.t2 - game.t1 + a2) / (a2 + 1.0)
        x2 = (game.t1 - game.t2 + 1.0) / (a2 + 1.0)
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)

        # Game with a = 5.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], a.subs(a, 5))
        x21 = (5 * game.t2 - 5 * game.t1 + a2) / (a2 + 1)
        x22 = (5 * game.t1 - 5 * game.t2 + 1) / (a2 + 1)
        game.calculate_equilibrium()
        self.assertEqual(x21, game.x1)
        self.assertEqual(x22, game.x2)

        # WIP: Game with a = 1 if p < split else 5.
        # split = 1 / 2
        # # NOTE: Regardless of having all integers in input, the resulting equilibrium is real.
        # game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], Piecewise((1, p < split), (5, p >= split)))
        # x1 = Piecewise((x11, p < split), (x21, p >= split))
        # x2 = Piecewise((x12, p < split), (x22, p >= split))
        # game.calculate_equilibrium()
        # self.assertEqual(x1, game.x1)
        # self.assertEqual(x2, game.x2)

        # Game with a = p.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], p)
        x1 = Piecewise(((game.t2 - game.t1 + a2) / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       (a2 / (game.t1 - game.t2 + a2 + 1), True))
        x2 = Piecewise((1 / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       ((game.t1 - game.t2 + 1) / (game.t1 - game.t2 + a2 + 1), True))
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)

        # Game with a = p + 1.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], p + 1)
        x1 = Piecewise(((2 * game.t2 - 2 * game.t1 + a2) / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       ((game.t2 - game.t1 + a2) / (game.t1 - game.t2 + a2 + 1), True))
        x2 = Piecewise(((1 - game.t2 + game.t1) / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       ((2 * game.t1 - 2 * game.t2 + 1) / (game.t1 - game.t2 + a2 + 1), True))
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)
