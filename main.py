from src import TwoLinkPricingGame

if __name__ == '__main__':
    game = TwoLinkPricingGame([[1, 0], [2, 0]])
    game.calculate_best_responses()
    game.analyse(stop=8)
