import unittest
from sympy import symbols, Piecewise, sqrt, Rational
from src import TwoLinkHeterogeneousGame, TwoLinkHeterogeneousPricingGame


class TestTwoLinkHeterogeneousGame(unittest.TestCase):

    def setUp(self):
        self.a, self.p = symbols('a p', nonnegative=True)

    def test_calculate_equilibrium(self):
        a2 = symbols('a2', nonnegative=True)

        # Game with a = 1.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], self.a.subs(self.a, 1.0))
        x1 = (game.t2 - game.t1 + a2) / (a2 + 1.0)
        x2 = (game.t1 - game.t2 + 1.0) / (a2 + 1.0)
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)

        # Game with a = 5.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], self.a.subs(self.a, 5))
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
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], self.p)
        x1 = Piecewise(((game.t2 - game.t1 + a2) / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       (a2 / (game.t1 - game.t2 + a2 + 1), True))
        x2 = Piecewise((1 / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       ((game.t1 - game.t2 + 1) / (game.t1 - game.t2 + a2 + 1), True))
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)

        # Game with a = p + 1.
        game = TwoLinkHeterogeneousGame([[1, 0], [a2, 0]], self.p + 1)
        x1 = Piecewise(((2 * game.t2 - 2 * game.t1 + a2) / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       ((game.t2 - game.t1 + a2) / (game.t1 - game.t2 + a2 + 1), True))
        x2 = Piecewise(((1 - game.t2 + game.t1) / (game.t2 - game.t1 + a2 + 1), game.t1 <= game.t2),
                       ((2 * game.t1 - 2 * game.t2 + 1) / (game.t1 - game.t2 + a2 + 1), True))
        game.calculate_equilibrium()
        self.assertEqual(x1, game.x1)
        self.assertEqual(x2, game.x2)

    def test_calculate_best_responses(self):
        a2 = symbols('a2', nonnegative=True)

        # Game with a = 1.
        game = TwoLinkHeterogeneousPricingGame([[1, 0], [a2, 0]], self.a.subs(self.a, 1.0))
        br1 = Piecewise((0.5 * (game.t2 + a2), game.t2 < a2 + 2.0), (game.t2 - 1.0, True))
        br2 = Piecewise((0.5 * (game.t1 + 1), game.t1 < 2.0 * a2 + 1.0), (game.t1 - a2, True))
        game.calculate_best_responses()
        self.assertEqual(br1, game.br1)
        self.assertEqual(br2, game.br2)

        # Game with a = 5.
        game = TwoLinkHeterogeneousPricingGame([[1, 0], [a2, 0]], self.a.subs(self.a, 5.0))
        br1 = Piecewise((0.5 * game.t2 + 0.1 * a2, game.t2 < 0.2 * a2 + 0.4), (game.t2 - 0.2, True))
        br2 = Piecewise((0.5 * game.t1 + 0.1, game.t1 < 0.4 * a2 + 0.2), (game.t1 - 0.2 * a2, True))
        game.calculate_best_responses()
        self.assertEqual(br1, game.br1)
        self.assertEqual(br2, game.br2)

        # Game with a = p + 1.
        game = TwoLinkHeterogeneousPricingGame([[1, 0], [2, 0]], self.p + 1)
        br1 = Piecewise(
            (sqrt(5 * (3 - game.t2)) - 3 + game.t2, game.t2 < Rational(6, 5)),
            (game.t2, game.t2 < Rational(3, 2)),
            (game.t2 + 3 - sqrt(2 * game.t2 + 6), game.t2 <= 5),
            (game.t2 - 1, game.t2 > 5)
        )
        num = sqrt(Rational(88, 81) - 16 * sqrt(10) / 81)
        br2 = Piecewise(
            (2 * sqrt(3 - game.t1) - 3 + game.t1, game.t1 <= num),
            (game.t1 + 3 - sqrt(10 * game.t1 + 30) / 2, game.t1 <= 7),
            (game.t1 - 2, game.t1 > 7)
        )
        game.calculate_best_responses()
        self.assertEqual(br1, game.br1)
        self.assertEqual(br2, game.br2)
