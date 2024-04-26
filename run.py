from hyper_optimizer.optimizer import HyperOptimizer

if __name__ == "__main__":
    h = HyperOptimizer(script_mode=True)
    h.start_search()
