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

        self.t1, self.t2 = symbols('t1 t2', real=True, nonnegative=True)
        self.c1 = self.l1 + self.t1
        self.c2 = self.l2 + self.t2

        self.br1 = None
        self.br2 = None

    def calculate_equilibrium(self):
        """Calculate the optimal flow x w.r.t. tolls (t1, t2) at which there is an equilibrium.
            Optimal flow occurs when the two links have the same cost.
            Use the expression c1 == c2, solve for x1 and store the resulting expression.
        """
        solution = solve(self.c1 - self.c2, self.x1)
        if len(solution) == 1:
            self.x1 = solution[0]
            self.x2 = simplify(1 - self.x1)
        elif len(solution) > 1:
            # TODO: First remove the solutions that are not in the range [0, 1].

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
        # br1_2 = self.t2 - self.l1.subs('x1', 1)
        # br2_2 = self.t1 - self.l2.subs('x1', 0)
        # More fancy way to calculate the best response, useful when child classes are using it.
        br1_2 = solve(self.c1.subs('x1', 1) - self.c2.subs(self.l2, 0), self.t1)[0]
        br2_2 = solve(self.c2.subs('x1', 0) - self.c1.subs(self.l1, 0), self.t2)[0]

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
            bt1, bt2 = self.br1.subs(self.t2, bt2).evalf(), self.br2.subs(self.t1, bt1).evalf()
        return bt1, bt2

    def plot_profit_functions(self, stop=10, step=0.1, ax1=None, ax2=None):
        """Plot the profit functions for each player.

        Args:
            stop (int): The stop value for the plot.
            step (float): The step value for the plot.
            ax1 (Axes): Optional. The first axes to plot on.
            ax2 (Axes): Optional. The second axes to plot on.

        Returns:
            list: The two axes for the plot.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.x1 in self.x1.free_symbols or self.x2 in self.x2.free_symbols:
            print("Calculating equilibrium...")
            self.calculate_equilibrium()
            print("Equilibrium calculated.")

        if ax1 is None or ax2 is None:
            _, [_ax1, _ax2] = plt.subplots(1, 2, figsize=(12, 6))
            ax1 = _ax1 if ax1 is None else ax1
            ax2 = _ax2 if ax2 is None else ax2
        t = np.linspace(0, stop, int(stop / step))

        ax1.set_title('Profit for Toll Owner 1')
        p1 = [self.x1.subs([(self.t2, stop / 2), (self.t1, i)]) * i for i in t]
        ax1.plot(t, p1)
        ax1.legend([f't2={stop / 2}'])

        ax2.set_title('Profit for Toll Owner 2')
        p2 = [self.x2.subs([(self.t1, stop / 2), (self.t2, i)]) * i for i in t]
        ax2.plot(t, p2)
        ax2.legend([f't1={stop / 2}'])

        return [ax1, ax2]

    def plot_best_responses(self, stop=10, step=0.1, ax=None):
        """Plot the best responses for each player.

        Args:
            stop (int): The stop value for the plot.
            step (float): The step value for the plot.
            ax (Axes): Optional. The axes to plot on.

        Returns:
            Axes: The axis for the plot.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.br1 is None or self.br2 is None:
            print("Calculating best responses...")
            self.calculate_best_responses()
            print("Best responses calculated.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.set_title('Best Responses')

        t2 = np.linspace(0, stop, int(stop / step))
        t1 = [self.br1.subs(self.t2, i) for i in t2]
        ax.plot(t1, t2)
        ax.set_xlabel('Toll Owner 1')

        t1 = np.linspace(0, stop, int(stop / step))
        t2 = [self.br2.subs(self.t1, i) for i in t1]
        ax.plot(t1, t2)
        ax.set_ylabel('Toll Owner 2')

        # Also add the approximate pricing equilibrium.
        equilibrium = self.approximate_pricing_equilibrium()
        ax.plot(equilibrium[0], equilibrium[1], 'ro', label='Equilibrium at ({0}, {1})'.format(
            round(equilibrium[0], 2), round(equilibrium[1], 2)
        ))

        ax.legend()
        return ax

    def print_latex(self):
        """Print the game in LaTeX format."""
        from sympy import latex

        print(f"\\begin{{align*}}")
        # print(f"\\text{{Player 1: }} & \\text{{Latency: }} {latex(self.l1)} \\\\")
        # print(f"\\text{{Player 2: }} & \\text{{Latency: }} {latex(self.l2)} \\\\")
        # print(f"\\text{{Player 1: }} & \\text{{Cost: }} {latex(self.c1)} \\\\")
        # print(f"\\text{{Player 2: }} & \\text{{Cost: }} {latex(self.c2)} \\\\")
        print(f"x_1 &= {latex(self.x1)} \\\\")
        print(f"x_2 &= {latex(self.x2)} \\\\")
        print(f"BR_1 &= {latex(self.br1)} \\\\") if self.br1 is not None else None
        print(f"BR_2 &= {latex(self.br2)} \\\\") if self.br2 is not None else None
        print(f"\\end{{align*}}")

    def analyse(self, stop=10, step=0.1):
        """Analyse the game.

        Args:
            stop (int): The stop value for the plot.
            step (float): The step value for the plot.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        gs = fig.add_gridspec(4, 4)
        ax1 = fig.add_subplot(gs[:2, :2])
        ax2 = fig.add_subplot(gs[:2, 2:])
        ax3 = fig.add_subplot(gs[2:, 1:3])

        print("Plotting profit functions...")
        self.plot_profit_functions(stop, step, ax1, ax2)
        print("Profit functions plotted.")

        print("Plotting best responses...")
        self.plot_best_responses(stop, step, ax3)
        print("Best responses plotted.")

        print("Printing the game in LaTeX format...")
        self.print_latex()
        print("LaTeX format printed.")

        plt.show()
