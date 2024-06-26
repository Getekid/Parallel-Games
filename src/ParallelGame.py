from sympy import symbols, solve, simplify


class TwoLinkParallelGame:
    """A class for the 2-link network parallel game."""

    def __init__(self, affine_latencies):
        """Initiates the game class.

        Args:
            affine_latencies (list): A list of the latency functions' factors.
        """
        self.x1 = symbols('x1')
        self.x2 = 1 - self.x1

        [[a1, b1], [a2, b2]] = affine_latencies
        self.l1 = a1 * self.x1 + b1
        self.l2 = a2 * self.x2 + b2

        self.t1, self.t2 = symbols('t1 t2')
        self.c1 = self.l1 + self.t1
        self.c2 = self.l2 + self.t2

        self.br1 = None
        self.br2 = None

    def calculate_equilibrium(self):
        """Calculate the optimal flow x w.r.t. tolls (t1, t2) at which there is an equilibrium.
            # Optimal flow occurs when the two links have the same cost.
            # Use the expression c1 == c2, solve for x1 and store the resulting expression.
        """
        solution = solve(self.c1 - self.c2, self.x1)
        if len(solution) == 1:
            self.x1 = solution[0]
            self.x2 = simplify(1 - self.x1)

    def calculate_best_responses(self):
        """Calculate the best responses for each player."""
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
        br1_2 = self.t2 - self.l1.subs('x1', 1)
        br2_2 = self.t1 - self.l2.subs('x1', 0)

        # Step 5: Combine the results.
        self.br1 = [
            {'cond': br1_1_cond, 'func': br1_1},
            {'cond': br1_2_cond, 'func': br1_2}
        ]
        self.br2 = [
            {'cond': br2_1_cond, 'func': br2_1},
            {'cond': br2_2_cond, 'func': br2_2}
        ]
