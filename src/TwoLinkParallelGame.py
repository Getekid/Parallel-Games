from sympy import symbols, solve, simplify, Piecewise, piecewise_fold, nan


class TwoLinkParallelGame:
    """A class for the 2-link parallel network (homogeneous) game."""

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

        self.t1, self.t2 = symbols('t1 t2', nonnegative=True)
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
        elif len(solution) > 1:
            # Combine the solutions by adding them and replacing nan with 0.
            solution = [sol.subs(nan, 0) for sol in solution]
            self.x1 = piecewise_fold(sum(solution))
            self.x2 = simplify(1 - self.x1)


class TwoLinkPricingGame(TwoLinkParallelGame):
    """A class for the 2-link parallel network (homogeneous) pricing game."""

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
        self.br1 = Piecewise((br1_1, br1_1_cond), (br1_2, br1_2_cond))
        self.br2 = Piecewise((br2_1, br2_1_cond), (br2_2, br2_2_cond))

    def approximate_pricing_equilibrium(self, max_iter=50, init=(0, 0)):
        """Approximate the pricing equilibrium.

        Args:
            max_iter (int): The maximum number of iterations.
            init (tuple): The initial values for the tolls.

        Returns:
            tuple: The approximate pricing equilibrium.
        """
        if self.br1 is None or self.br2 is None:
            return None

        bt1, bt2 = init
        for _ in range(max_iter):
            bt1, bt2 = self.br1.subs(self.t2, bt2), self.br2.subs(self.t1, bt1)
        return bt1, bt2

    def plot_best_reponses(self, stop=10, step=0.1):
        """Plot the best responses for each player.

        Args:
            stop (int): The stop value for the plot.
            step (float): The step value for the plot.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        y1 = np.linspace(0, stop, int(stop / step))
        x1 = [self.br1.subs(self.t2, i) for i in y1]
        plt.plot(x1, y1)
        plt.xlabel('Toll Owner 1')

        x2 = np.linspace(0, stop, int(stop / step))
        y2 = [self.br2.subs(self.t1, i) for i in x2]
        plt.plot(x2, y2)
        plt.ylabel('Toll Owner 2')

        # Also add the approximate pricing equilibrium.
        equilibrium = self.approximate_pricing_equilibrium()
        plt.plot(equilibrium[0], equilibrium[1], 'ro', label='Equilibrium at ({0}, {1})'.format(
            round(equilibrium[0], 2), round(equilibrium[1], 2)
        ))

        plt.legend()
        plt.show()
