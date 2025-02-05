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
