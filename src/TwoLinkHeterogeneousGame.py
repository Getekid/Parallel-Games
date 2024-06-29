from sympy import symbols, Piecewise, piecewise_fold, solve, nan, simplify, Union, Intersection, Interval, S
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
        self.c1 = piecewise_fold(self.l1 + self.a_s * self.t1)
        self.c2 = piecewise_fold(self.l2 + self.a_s * self.t2)


class TwoLinkHeterogeneousPricingGame(TwoLinkHeterogeneousGame, TwoLinkPricingGame):
    """A class for the 2-link parallel network (heterogeneous) pricing game."""

    def calculate_best_responses(self):
        """Calculate the best responses for each player.
            Currently same implementation as the TwoLinkPricingGame, instead of Step 4
        """
        # If the time-money sensitivity function is not piecewise,
        # the game is equivalent to a TwoLinkPricingGame, so use that implementation.
        if not self.a_s.is_Piecewise:
            super().calculate_best_responses()
            return

        # Step 1: Calculate the optimal flow x at which there is an equilibrium.
        self.calculate_equilibrium()

        # Step 2: Calculate the argmax for each profit function (differentiate, find root and solve for t).
        pr1 = self.x1 * self.t1
        pr2 = self.x2 * self.t2
        br1 = solve(pr1.diff(self.t1), self.t1)
        br2 = solve(pr2.diff(self.t2), self.t2)
        if len(br1) == 0 or len(br2) == 0:
            return

        # TODO: All further steps are duplicated for player 1 and 2, consider refactoring.

        # Step 3: Prepare the Piecewise functions for the best responses, handle complements and intersections.

        # Since we know a_s is Piecewise, there will be multiple Piecewise solutions above.
        # Each element of br1 and br2 is a Piecewise function with a single (expr, cond) tuple and a (nan, True).
        # Combine them by taking the first element from each Piecewise solution to form a single Piecewise function.
        br1 = [br.as_expr_set_pairs()[0] for br in br1 if br.as_expr_set_pairs()[0][0] != nan]
        br1_intervals = [pair[1] for pair in br1 if pair[1] != S.Reals]
        br2 = [br.as_expr_set_pairs()[0] for br in br2 if br.as_expr_set_pairs()[0][0] != nan]
        br2_intervals = [pair[1] for pair in br2 if pair[1] != S.Reals]

        # If the intervals don't cover all [0, Inf), add the missing interval.
        complement1 = Union(*br1_intervals).complement(Interval(0, S.Infinity))
        if complement1 != S.EmptySet:
            br1.append((self.t2, complement1))
        complement2 = Union(*br2_intervals).complement(Interval(0, S.Infinity))
        if complement2 != S.EmptySet:
            br2.append((self.t1, complement2))

        # Order by interval, first the lower bound and then the higher one.
        br1.sort(key=lambda pair: (pair[1].start, pair[1].end))
        br2.sort(key=lambda pair: (pair[1].start, pair[1].end))

        # If any two intervals overlap, it will be consecutive ones.
        # Find where the two expressions are equal and split the interval there.
        # I should have only one solution in the intersection interval.
        for i in range(len(br1) - 1):
            intersection1 = Intersection(br1[i][1], br1[i + 1][1])
            if intersection1 == S.EmptySet:
                continue
            solutions = solve(pr1.subs(self.t1, br1[i][0]) - pr1.subs(self.t1, br1[i + 1][0]), self.t2)
            for solution in solutions:
                if intersection1.contains(solution):
                    br1[i] = (br1[i][0], Interval(br1[i][1].start, solution))
                    br1[i + 1] = (br1[i + 1][0], Interval.open(solution, br1[i + 1][1].end))
                    break
        for i in range(len(br2) - 1):
            intersection2 = Intersection(br2[i][1], br2[i + 1][1])
            if intersection2 == S.EmptySet:
                continue
            solutions = solve(pr2.subs(self.t2, br2[i][0]) - pr2.subs(self.t2, br2[i + 1][0]), self.t1)
            for solution in solutions:
                if intersection2.contains(solution):
                    br2[i] = (br2[i][0], Interval(br2[i][1].start, solution))
                    br2[i + 1] = (br2[i + 1][0], Interval.open(solution, br2[i + 1][1].end))
                    break

        # Step 4: Calculate the limit and best response of the above argmax,
        # i.e. the argmax and best response results when the player is controlling all the flow.

        # For the limit use the function element on the highest interval, i.e. the last one in the list.
        br1_limit = solve(self.x1.subs(self.t1, br1[-1][0]) - 1, self.t2)
        br2_limit = solve(self.x2.subs(self.t2, br2[-1][0]) - 1, self.t1)
        if len(br1_limit) != 1 or len(br2_limit) != 1:
            return
        br1_limit, br2_limit = br1_limit[0], br2_limit[0]

        # For the max solve for the cost to be equal when a player controls all the flow.
        # c1 and c2 are Piecewise functions, first element when t1 > t2 (true when x1 = 0)
        # and second when t1 <= t2 (true when x1 = 1).
        br1_max = solve(self.c1.args[1].expr.subs('x1', 1) - self.c2.args[1].expr.subs('x1', 1), self.t1)[0]
        br2_max = solve(self.c2.args[0].expr.subs('x1', 0) - self.c1.args[0].expr.subs('x1', 0), self.t2)[0]

        # Update the best response lists.
        br1[-1] = (br1[-1][0], Interval(br1[-1][1].start, br1_limit))
        br1.append((br1_max, Interval.open(br1_limit, S.Infinity)))
        br2[-1] = (br2[-1][0], Interval(br2[-1][1].start, br2_limit))
        br2.append((br2_max, Interval.open(br2_limit, S.Infinity)))

        # Step 5: Convert the intervals back to relational expressions
        # and combine the results to a single Piecewise function.
        br1 = [(pair[0], pair[1].as_relational(self.t2)) for pair in br1]
        br2 = [(pair[0], pair[1].as_relational(self.t1)) for pair in br2]
        self.br1 = simplify(Piecewise(*br1))
        self.br2 = simplify(Piecewise(*br2))
