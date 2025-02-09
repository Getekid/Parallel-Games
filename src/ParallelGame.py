import numpy as np


class ParallelGame:
    """n-link Parallel Game class.
        TODO: Add random seed.

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

    def appr_pricing_equilibrium(self, tolls_init=None, n_rounds=100, n_samples=100):
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
                costs_all_flow = a + b + tolls
                # l_i(0) + Max toll = max c_j.
                max_toll = np.max(costs_all_flow) - b[link]

                # Generate two sets of samples.
                # 1. Uniform over the toll min and max value.
                # 2. Normal around the toll's current value.
                rng = np.random.default_rng()
                toll_samples = np.concatenate((rng.uniform(0, max_toll, n_samples),
                                               rng.normal(tolls[link], 1 / (max_toll + r), n_samples)))
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
            get_flow (tolls=None): Returns the Nash Equilibrium flow between links for given tolls.
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
            (np.array): Flow Nash equilibrium.
        """
        tolls = np.array(tolls) if tolls is not None else np.zeros(self.n)
        if np.all(tolls == tolls[0]) or self.n == 1:
            return super().get_flow(tolls)

        a = self.latencies[:, 0]
        b = self.latencies[:, 1]

        # Solve a(p) = (a_j * x_j + b_j - a_i * x_i - b_i) / t_i - t_j
        # for i, j links consecutive according to toll ordering descending.
        tolls_desc_indices = tolls.argsort()[::-1]
        tolls_desc_next_indices = np.roll(tolls_desc_indices, -1)
        toll_diffs = tolls[tolls_desc_indices] - tolls[tolls_desc_next_indices]
        # For dividing with toll_diffs, replace 0 with 1 (to ignore this case).
        toll_diffs_div = toll_diffs.copy()
        toll_diffs_div[toll_diffs == 0] = 1

        flow_factors = np.zeros((self.n, self.n))
        constants = np.zeros(self.n)
        flow_factors[tolls_desc_indices, tolls_desc_indices] = a[tolls_desc_indices] / toll_diffs_div
        flow_factors[tolls_desc_indices, tolls_desc_next_indices] = -1 * a[tolls_desc_next_indices] / toll_diffs_div
        constants[tolls_desc_indices] = (b[tolls_desc_next_indices] - b[tolls_desc_indices]) / toll_diffs_div

        # a(p) is linear so a(p) = a * p + b for scalar and constant factors a, b.
        # First (ordered) link will have a(p) = a * x_1 + b, second one a(p) = a * (x_1 + x_2) + b, etc.
        # So add the dist scalar factor to the lower triangle of the ordered flow matrix.
        tril_indices = np.tril_indices(self.n)
        tril_tolls_desc_indices = (tolls_desc_indices[tril_indices[0]], tolls_desc_indices[tril_indices[1]])
        #  Ignore when toll diff is 0.
        toll_diffs_zero = tolls_desc_indices[toll_diffs == 0]
        if len(toll_diffs_zero) > 0:
            tril_zero = np.any(tril_tolls_desc_indices[0] == toll_diffs_zero[:, None], axis=0)
            tril_tolls_desc_indices = (tril_tolls_desc_indices[0][~tril_zero], tril_tolls_desc_indices[1][~tril_zero])
        flow_factors[tril_tolls_desc_indices] += self.dist[0]
        # Also subtract the constant factor from the constants. Ignore when toll diff is 0.
        constants[tolls_desc_indices] -= np.repeat(self.dist[1], self.n) * (toll_diffs != 0)

        # Fill in the last row with x1+x2+...+xn = 1.
        flow_factors[tolls_desc_indices[-1]] = np.ones(self.n)
        constants[tolls_desc_indices[-1]] = 1
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


class StepDistParallelGame(ParallelGame):
    """n-link Heterogeneous Parallel Game class with step distribution function.

        Attributes:
            n (int): The number of links in the parallel network.
            latencies (np.array): A list of factors for the latencies of each link.
            dist_values (list|np.array): A list of increasing values for the step distribution function.
            dist_cond (list|np.array): A list of increasing numbers in (0, 1) at which
                the step distribution function changes values. Has size -1 of dist_values

        Methods:
            get_flow (tolls=None): Returns the Nash Equilibrium flow between links for given tolls.
            get_pricing_equilibrium: Calculates and returns the Nash Equilibrium for the pricing game on the parallel network.
            get_2_link_pricing_equilibrium: Calculates and returns the Nash Equilibrium
                for the pricing game on a 2-link parallel network.
        """

    def __init__(self, latencies, dist_values=None, dist_cond=None):
        """Initiates the game class.

        Args:
           latencies (list|np.array): An [a, b] list of the (affine) latency functions' factors.
           dist_values (list|np.array): A list of increasing values for the step distribution function.
           dist_cond (list|np.array): A list of increasing numbers in (0, 1) at which
                the step distribution function changes values. Has size -1 of dist_values
        """
        super().__init__(latencies)
        self.dist_values = np.array(dist_values) if dist_values is not None else np.array([0, 1])
        self.dist_cond = np.array(dist_cond) if dist_cond is not None else np.linspace(0, 1, len(dist_values) + 1)[1:-1]

    def get_step_value(self, player):
        """Returns the step distribution function value for the given player.

        Args:
            player (Player): The player whose step distribution value to return.

        Returns:
            (np.float): The step distribution function value for the given player.
        """
        player = np.max((player, 0))
        player = np.min((player, 1))
        cond = player < self.dist_cond
        return self.dist_values[-1 if not cond.any() else cond.argmax()]

    def get_flow(self, tolls=None):
        """Returns the flow Nash equilibrium for a given set of tolls.
            Implemented only for 2-link networks.

        Args:
            tolls (list|np.array): List of tolls to use.

        Returns:
            (np.array): Flow Nash equilibrium.
        """
        tolls = np.array(tolls) if tolls is not None else np.zeros(self.n)
        if np.all(tolls == tolls[0]) or self.n == 1:
            return super().get_flow(tolls)
        if self.n > 2:
            raise "This method is not implemented for n > 2."

        # The flow will correspond to a single step value, so search for it by applying the tolls to the
        # respective pseudo-heterogeneous (tolls * ak) and keeping the one the flow matches (should be unique).
        for k, ak in enumerate(self.dist_values):
            flow = super().get_flow(ak * tolls)
            flow_lower = flow[0] if tolls[0] > tolls[1] else flow[1]
            if self.get_step_value(flow_lower) == ak:
                return flow

    def get_2_link_pricing_equilibrium(self):
        """Calculates and returns the Nash Equilibrium
            for the pricing game on a 2-link parallel network.

        Returns:
            (np.array|bool): The tolls that are the Nash equilibrium
                for the 2-link pricing game, if it exists, or False otherwise.
        """
        if self.n != 2:
            raise "This Algorithm is valid only for 2-link parallel games."
        a1, a2 = self.latencies[:, 0]
        b1, b2 = self.latencies[:, 1]

        # Helper functions.
        def toll_diff(a, x1):
            return (1 / a) * ((a1 + a2) * x1 - a2 + b1 - b2)
        def profit_1(a, t1, t2):
            return (t1 / (a1 + a2)) * (a2 + b2 - b1 + a * (t2 - t1))
        def profit_2(a, t1, t2):
            return (t2 / (a1 + a2)) * (a1 + b1 - b2 + a * (t1 - t2))
        def best_response_1(a, t2):
            if t2 <= (1 / a) * (2 * a1 + a2 + b1 - b2):
                return t2 / 2 + (1 / (2 * a)) * (a2 + b2 - b1)
            else:
                return t2 - (1 / a) * (a1 + b1 - b2)
        def best_response_2(a, t1):
            if t1 <= (1 / a) * (2 * a2 + a1 + b2 - b1):
                return t1 / 2 + (1 / (2 * a)) * (a1 + b1 - b2)
            else:
                return t1 - (1 / a) * (a2 + b2 - b1)

        # Algorithm.
        t1_cand, t2_cand = (1 / 3) * (2 * a2 + a1 + b2 - b1), (1 / 3) * (2 * a1 + a2 + b1 - b2)
        x1_ne, x2_ne = t1_cand / (a1 + a2), t2_cand / (a1 + a2)
        split = self.get_step_value(x1_ne) if t1_cand > t2_cand else self.get_step_value(x2_ne)
        t1_cand, t2_cand = t1_cand / split, t2_cand / split
        pr1, pr2 = x1_ne * t1_cand, x2_ne * t2_cand

        for k, ak in enumerate(self.dist_values):
            if ak == split:
                continue
            t1k, t2k = best_response_1(ak, t2_cand), best_response_2(ak, t1_cand)
            p_low = self.dist_cond[k - 1] if k != 0 else 0
            p_high = self.dist_cond[k] if k != len(self.dist_values) else 1
            t_low_low, t_low_high = toll_diff(ak, p_low), toll_diff(ak, p_high)
            t_high_low, t_high_high = toll_diff(ak, 1 - p_high), toll_diff(ak, 1 - p_low)
            t1k_low_low, t1k_low_high = t2_cand - t_low_low, t2_cand - t_low_high
            t1k_high_low, t1k_high_high = t2_cand - t_high_low, t2_cand - t_high_high
            t2k_low_low, t2k_low_high = t1_cand + t_low_low, t1_cand + t_low_high
            t2k_high_low, t2k_high_high = t1_cand + t_high_low, t1_cand + t_high_high

            if t1k_low_low <= t1k <= t1k_low_high or t1k_high_low <= t1k <= t1k_high_high:
                pr1k = profit_1(ak, t1k, t2_cand)
            else:
                pr1k = np.max((profit_1(ak, t1k_low_low, t2_cand), profit_1(ak, t1k_low_high, t2_cand),
                               profit_1(ak, t1k_high_low, t2_cand), profit_1(ak, t1k_high_high, t2_cand)))
            if t2k_low_low <= t2k <= t2k_low_high or t2k_high_low <= t2k <= t2k_high_high:
                pr2k = profit_2(ak, t2k, t1_cand)
            else:
                pr2k = np.max((profit_2(ak, t1_cand, t2k_low_low), profit_2(ak, t1_cand, t2k_low_high),
                               profit_2(ak, t1_cand, t2k_high_low), profit_2(ak, t1_cand, t2k_high_high)))

            if pr1k > pr1 or pr2k > pr2:
                return False

        return np.array([t1_cand, t2_cand])
