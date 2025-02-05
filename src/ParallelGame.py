import numpy as np


class ParallelGame:
    """n-link Parallel Game class.

    Attributes:
        n (int): The number of links in the parallel network.
        latencies (np.array): A list of factors for the latencies of each link.

    Methods:
        get_flow (tolls=None): Returns the Wardrop flow between links for given tolls.
        get_pricing_equilibrium: Calculates and returns the Nash Equilibrium for the pricing game on the parallel network.
    """

    def __init__(self, latencies):
        """Initiates the game class.

       Args:
           latencies (list|np.array): An [a, b] list of the (affine) latency functions' factors.
       """
        self.n = len(latencies)
        self.latencies = np.array(latencies)

    def get_flow(self, tolls=None):
        """Returns the flow Wardrop equilibrium for given set of tolls.

        Args:
            tolls (list|np.array): List of tolls to use.

        Returns:
            (np.array): Flow Wardrop equilibrium.
        """
        tolls = np.array(tolls) if tolls is not None else np.zeros(self.n)

        flow = np.zeros(self.n)
        for i in range(self.n):
            num = 1 + np.array([(self.latencies[j][1] + tolls[j] - self.latencies[i][1] - tolls[i]) / self.latencies[j][0] for j in range(self.n)]).sum()
            den = np.array([self.latencies[i][0] / self.latencies[j][0] for j in range(self.n)]).sum()
            flow[i] = num / den

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
        # Default factor for t_i is -1/a_i.
        toll_factors = np.tile(-1 / self.latencies[:, 0], (self.n, 1))

        # Factor for t_i in i-th row is 2 * sum(1/a_j when j != i).
        inv_a = 1 / self.latencies[:, 0]
        inv_a = inv_a.sum() - inv_a
        np.fill_diagonal(toll_factors, 2 * inv_a)

        # Constants are 1 + sum((b_j-b_i)/a_j when j != i).
        b_a = self.latencies[:, 1] / self.latencies[:, 0]
        b_a = b_a.sum() - b_a
        constants = 1 + b_a - self.latencies[:, 1] * inv_a

        # Solve the system of linear equations and return the result.
        return np.linalg.solve(toll_factors, constants)
