from sympy import symbols, Piecewise, solve, false, simplify
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
        # If the time-money sensitivity function is not piecewise,
        # the game is equivalent to a TwoLinkPricingGame, so use that implementation.
        if not self.a_s.is_Piecewise:
            super().calculate_best_responses()
            return

        # Step 1: Calculate the optimal flow x at which there is an equilibrium.
        self.calculate_equilibrium()

        # Step 2: Calculate the argmax for each profit function (differentiate, find root and solve for t).
        p1 = self.x1 * self.t1
        p2 = self.x2 * self.t2
        br1 = solve(p1.diff(self.t1), self.t1)
        br2 = solve(p2.diff(self.t2), self.t2)
        if len(br1) == 0 or len(br2) == 0:
            return

        # Since we know a_s is Piecewise, there will be multiple Piecewise solutions above.
        # Each element of br1, br2 is a Piecewise function with a single (expr, cond) tuple and a (nan, True).
        # Combine them by taking the first element from each Piecewise solution to form a single Piecewise function.
        # Also solve the conditions to simplify them and reverse the order to make the lower conditions first.
        br1 = [(br.args[0].expr, solve(br.args[0].cond, self.t2)) for br in reversed(br1)]
        for i in range(len(br1)):
            if br1[i][1] == false:
                cond = self.t2 > 0
                if i != len(br1) - 1 and br1[i + 1][1] != false and br1[i + 1][1].__class__.__name__ in ['LessThan', 'StrictLessThan']:
                    cond = br1[i + 1][1].negated
                br1[i] = (self.t2, cond)
        br2 = [(br.args[0].expr, solve(br.args[0].cond, self.t1)) for br in reversed(br2)]
        for i in range(len(br2)):
            if br2[i][1] == false:
                cond = self.t1 > 0
                if i != len(br1) - 1 and br2[i + 1][1] != false and br2[i + 1][1].__class__.__name__ in ['LessThan', 'StrictLessThan']:
                    cond = br2[i + 1][1].negated
                br2[i] = (self.t1, cond)

        # Step 3: Calculate the limit of the above argmax,
        # i.e. the argmax results in the player controlling all the flow.
        # Use the Piecewise function element with the highest condition, i.e. the last one.
        br1_max_cond = solve(self.x1.subs(self.t1, br1[-1][0]) - 1, self.t2)
        br2_max_cond = solve(self.x2.subs(self.t2, br2[-1][0]) - 1, self.t1)
        if len(br1_max_cond) != 1 or len(br2_max_cond) != 1:
            return
        # br1_1_cond, br1_2_cond = self.t2 < br1_cond[0], self.t2 >= br1_cond[0]
        # br2_1_cond, br2_2_cond = self.t1 < br2_cond[0], self.t1 >= br2_cond[0]

        # Step 4: Calculate the best response when the player controls all the flow.
        # br1_2 = solve(self.c1.subs('x1', 1) - self.c2.subs(self.l2, 0), self.t1)[0]
        # br2_2 = solve(self.c2.subs('x1', 0) - self.c1.subs(self.l1, 0), self.t2)[0]
        br1_max = (self.a_s.args[1].expr.subs('x1', 1) * self.t2 - self.l1.subs('x1', 1)) / self.a_s.args[1].expr.subs('x1', 1)
        br2_max = (self.a_s.args[0].expr.subs('x1', 0) * self.t1 - self.l2.subs('x1', 0)) / self.a_s.args[0].expr.subs('x1', 0)

        br1[-1] = (br1[-1][0], self.t2 <= br1_max_cond[0])
        br1.append((br1_max, self.t2 > br1_max_cond[0]))
        br2[-1] = (br2[-1][0], self.t1 <= br2_max_cond[0])
        br2.append((br2_max, self.t1 > br2_max_cond[0]))

        # Step 5: Combine the results to a single Piecewise function.
        self.br1 = simplify(Piecewise(*br1))
        self.br2 = simplify(Piecewise(*br2))
