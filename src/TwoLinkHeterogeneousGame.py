from sympy import symbols, Piecewise
from src import TwoLinkParallelGame


class TwoLinkHeterogeneousGame(TwoLinkParallelGame):
    """A class for the 2-link parallel network (heterogeneous) pricing game."""

    def __init__(self, affine_latencies, a):
        """Initiates the game class.

        Args:
            affine_latencies (list): A list of the latency functions' factors.
            a (Expr): The time-money sensitivity function.
        """
        super().__init__(affine_latencies)

        # Define the time-money sensitivity function which takes a player [0, 1] as input,
        # i.e. it's an expression containing the symbol p.
        p = symbols('p', nonnegative=True)
        a_s = Piecewise((a.subs(p, self.x1), self.t1 > self.t2), (a.subs(p, self.x2), self.t1 <= self.t2))
        self.c1 = self.l1 + a_s * self.t1
        self.c2 = self.l2 + a_s * self.t2
