from sympy import symbols
from src import TwoLinkHeterogeneousPricingGame

if __name__ == '__main__':
    p = symbols('p', nonnegative=True)
    print("Initialising the game...")
    game = TwoLinkHeterogeneousPricingGame([[1, 0], [5, 0]], p + 1)
    print("Initialisation complete.")
    print("Analysing the game...")
    game.analyse(stop=5, step=0.001)
    print("Analysis complete.")
