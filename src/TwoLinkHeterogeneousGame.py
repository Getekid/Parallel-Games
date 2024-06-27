from sympy import symbols, Piecewise, solve
from src import TwoLinkParallelGame, TwoLinkPricingGame


class TwoLinkHeterogeneousGame(TwoLinkParallelGame):
    """A class for the 2-link parallel network (heterogeneous) game."""

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
        self.a_s = Piecewise((a.subs(p, self.x1), self.t1 > self.t2), (a.subs(p, self.x2), self.t1 <= self.t2))
        self.c1 = self.l1 + self.a_s * self.t1
        self.c2 = self.l2 + self.a_s * self.t2


class TwoLinkHeterogeneousPricingGame(TwoLinkHeterogeneousGame, TwoLinkPricingGame):
    """A class for the 2-link parallel network (heterogeneous) pricing game."""

    def calculate_best_responses(self):
        """Calculate the best responses for each player.
            Currently same implementation as the TwoLinkPricingGame, instead of Step 4
        """
        # TODO: Implement the method.
        # Step 1: Calculate the optimal flow x at which there is an equilibrium.
        self.calculate_equilibrium()

        # Step 2: Calculate the argmax for each profit function (differentiate, find root and solve for t).
        p1 = self.x1 * self.t1
        p2 = self.x2 * self.t2
        br1_1 = solve(p1.diff(self.t1), self.t1)
        br2_1 = solve(p2.diff(self.t2), self.t2)
        if len(br1_1) != 1 or len(br2_1) != 1:
            return
        br1_1, br2_1 = br1_1[0], br2_1[0]

        # Step 3: Calculate the limit of the above argmax,
        # i.e. the argmax results in the player controlling all the flow.
        br1_1_cond = solve(self.x1.subs(self.t1, br1_1) - 1, self.t2)
        br2_1_cond = solve(self.x2.subs(self.t2, br2_1) - 1, self.t1)
        if len(br1_1_cond) != 1 or len(br2_1_cond) != 1:
            return
        br1_1_cond, br1_2_cond = self.t2 < br1_1_cond[0], self.t2 >= br1_1_cond[0]
        br2_1_cond, br2_2_cond = self.t1 < br2_1_cond[0], self.t1 >= br2_1_cond[0]

        # Step 4: Calculate the best response when the player controls all the flow.
        br1_2 = (self.a_s * self.t2 - self.l1.subs('x1', 1)) / self.a_s
        br2_2 = (self.a_s * self.t1 - self.l2.subs('x1', 0)) / self.a_s

        # Step 5: Combine the results.
        self.br1 = Piecewise((br1_1, br1_1_cond), (br1_2, br1_2_cond))
        self.br2 = Piecewise((br2_1, br2_1_cond), (br2_2, br2_2_cond))
