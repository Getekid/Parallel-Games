import numpy as np


class ParallelGame:
    """n-link Parallel Game class.

    Attributes:
        n (int): The number of links in the parallel network.
        latencies (np.array): A list of factors for the latencies of each link.

    Methods:
        get_flow (tolls=None): Returns the Wardrop flow between links for given tolls.
        get_pricing_equilibrium: Calculates and returns the Nash Equilibrium for the pricing game on the parallel network.
        appr_pricing_equilibrium (tolls_init=None, n_rounds=1000, n_samples=100): Calculates and returns
            the approximate Nash Equilibrium for the pricing competition game on the parallel network.
    """

    def __init__(self, latencies):
        """Initiates the game class.

        Args:
           latencies (list|np.array): An [a, b] list of the (affine) latency functions' factors.
        """
        self.n = len(latencies)
        self.latencies = np.array(latencies)

    def get_flow(self, tolls=None):
        """Returns the flow Wardrop equilibrium for a given set of tolls.

        Args:
            tolls (list|np.array): List of tolls to use.

        Returns:
            (np.array): Flow Wardrop equilibrium.
        """
        tolls = np.array(tolls) if tolls is not None else np.zeros(self.n)
        a = self.latencies[:, 0]
        b = self.latencies[:, 1]

        # Using formula x_i(t) = (1 + sum((b_j+t_j-b_i-t_i)/a_j for all j)) / sum(a_i/a_j for all j).
        inv_a = 1 / a
        num = 1 + ((b + tolls) * inv_a).sum() - (b + tolls) * inv_a.sum()
        den = a * inv_a.sum()
        flow = num / den

        # If any of the flow elements are negative, remove them and re-calculate.
        # Do so by assuming a sub-game with only the non-negative flow links.
        flow_neg = flow < 0
        if flow_neg.sum() > 0:
            sub_game = self.__class__(self.latencies[~flow_neg])
            flow[~flow_neg] = sub_game.get_flow(tolls[~flow_neg])
            flow[flow_neg] = 0

        return flow

    def get_pricing_equilibrium(self):
        """Calculates and returns the Nash Equilibrium for the pricing competition game on the parallel network.

        Returns:
            (np.array): The set of tolls that are the Nash Equilibrium for the pricing game.
        """
        a = self.latencies[:, 0]
        b = self.latencies[:, 1]

        # Default factor for t_i is -1/a_i.
        toll_factors = np.tile(-1 / a, (self.n, 1))

        # Factor for t_i in i-th row is 2 * sum(1/a_j when j != i).
        inv_a = 1 / a
        inv_a_comp_sum = inv_a.sum() - inv_a  # Each element is the sum of the remaining ones.
        np.fill_diagonal(toll_factors, 2 * inv_a_comp_sum)

        # Constants are 1 + sum((b_j-b_i)/a_j when j != i).
        b_a = b / a
        b_a_comp_sum = b_a.sum() - b_a  # Each element is the sum of the remaining ones.
        constants = 1 + b_a_comp_sum - b * inv_a_comp_sum

        # Solve the system of linear equations and return the result.
        return np.linalg.solve(toll_factors, constants)

    def appr_pricing_equilibrium(self, tolls_init=None, n_rounds=1000, n_samples=100):
        """Calculates and returns the approximate Nash Equilibrium
            for the pricing competition game on the parallel network.

        Args:
            tolls_init (list|np.array): List of tolls to use in the first round.
            n_rounds (int): The number of rounds to simulate. In each round,
                all link operators take turns and change their toll to their best response one.
            n_samples (int): The number of samples to check for profit.
                Link operators in their turn will take samples to find a higher profit toll.

        Returns:
            (np.array): An list of the tolls played in each round.
                If an equilibrium exists, the tolls sequence should converge to it.
        """
        a = self.latencies[:, 0]
        b = self.latencies[:, 1]
        tolls_rounds = np.zeros((n_rounds, self.n))
        tolls = np.array(tolls_init) if tolls_init is not None else np.zeros(self.n)

        for r in range(n_rounds):
            for link in range(self.n):
                flow = self.get_flow(tolls)
                profits = flow * tolls
                costs = a * flow + b + tolls
                # l_i(0) + Max toll = max c_j.
                max_toll = np.max(costs) - b[link]

                # Generate two sets of samples.
                # 1. Uniform over the toll min and max value.
                # 2. Normal around the toll's current value.
                rng = np.random.default_rng()
                toll_samples = np.concatenate((rng.uniform(0, max_toll, n_samples),
                                               rng.normal(tolls[link], 1 / r, n_samples)))
                toll_samples[toll_samples < 0] *= -1

                tolls_sample = tolls.copy()
                for toll_sample in toll_samples:
                    tolls_sample[link] = toll_sample
                    profit_sample = self.get_flow(tolls_sample) * toll_sample
                    if profits[link] < profit_sample[link]:
                        profits[link] = profit_sample[link]
                        tolls[link] = toll_sample

            tolls_rounds[r] = tolls

        return tolls_rounds


class LinDistParallelGame(ParallelGame):
    """n-link Heterogeneous Parallel Game class with linear distribution function.

        Attributes:
            n (int): The number of links in the parallel network.
            latencies (np.array): A list of factors for the latencies of each link.
            dist (np.array): A list of factors for the linear distribution function.

        Methods:
            get_flow (tolls=None): Returns the Wardrop flow between links for given tolls.
            get_pricing_equilibrium: Calculates and returns the Nash Equilibrium for the pricing game on the parallel network.
        """

    def __init__(self, latencies, dist=None):
        """Initiates the game class.

        Args:
           latencies (list|np.array): An [a, b] list of the (affine) latency functions' factors.
           dist (list|np.array): An [a, b] list of the linear distribution function's factors.
        """
        super().__init__(latencies)
        self.dist = np.array(dist) if dist is not None else np.array([0, 1])

    def get_flow(self, tolls=None):
        """Returns the flow Nash equilibrium for a given set of tolls.

        Args:
            tolls (list|np.array): List of tolls to use.

        Returns:
            (np.array): Flow Wardrop equilibrium.
        """
        if tolls is None or np.array_equal(tolls, np.zeros(self.n)) or self.n == 1:
            return super().get_flow(tolls)

        tolls = np.array(tolls)
        a = self.latencies[:, 0]
        b = self.latencies[:, 1]

        # Solve a(p) = (a_j * x_j + b_j - a_i * x_i - b_i) / t_i - t_j
        # for i, j links consecutive according to toll ordering descending.
        tolls_desc_indices = tolls.argsort()[::-1]
        tolls_desc_next_indices = np.roll(tolls_desc_indices, -1)
        toll_diffs = tolls[tolls_desc_indices] - tolls[tolls_desc_next_indices]

        flow_factors = np.zeros((self.n, self.n))
        flow_factors[tolls_desc_indices, tolls_desc_indices] = a[tolls_desc_indices] / toll_diffs
        flow_factors[tolls_desc_indices, tolls_desc_next_indices] = -1 * a[tolls_desc_next_indices] / toll_diffs
        constants = (b[tolls_desc_next_indices] - b[tolls_desc_indices]) / toll_diffs

        # a(p) is linear so a(p) = a * p + b for scalar and constant factors a, b.
        # First (ordered) link will have a(p) = a * x_1 + b, second one a(p) = a * (x_1 + x_2) + b, etc.
        # So add the dist scalar factor to the lower triangle of the ordered flow matrix.
        tril_indices = np.tril_indices(self.n)
        flow_factors[tolls_desc_indices[tril_indices[0]], tolls_desc_indices[tril_indices[1]]] += self.dist[0]
        # Also subtract the constant factor from the constants.
        constants -= self.dist[1]

        # Fill in the last row with x1+x2+...+xn = 1.
        flow_factors[-1] = np.ones(self.n)
        constants[-1] = 1
        # Solve the system of linear equations
        flow = np.linalg.solve(flow_factors, constants)

        # If any of the flow elements are negative, remove them and re-calculate.
        # Do so by assuming a sub-game with only the non-negative flow links.
        flow_neg = flow < 0
        if flow_neg.sum() > 0:
            sub_game = self.__class__(self.latencies[~flow_neg])
            flow[~flow_neg] = sub_game.get_flow(tolls[~flow_neg])
            flow[flow_neg] = 0

        return flow
