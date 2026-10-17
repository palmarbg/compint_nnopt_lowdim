import torch

class GeneticAlgorithmOptimizer:
    def __init__(self, population, dim, fitness_fn, device):
        self.population = population
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.device = device

    def train(self, selection_fn, crossover_fn, mutation_fn, max_iter=50, patience=10):
        genomes = torch.rand((self.population, self.dim), device=self.device) * 2 - 1
                
        best_genom = torch.zeros((self.dim, ), device=self.device)
        best_fitness = -torch.inf

        patience_counter = 0
        fitness_history = []

        for i in range(max_iter):
            # evaluate fitness
            fitness = self.fitness_fn(genomes).to(self.device)
            
            best_idx = torch.argmax(fitness).item()
            if best_fitness < fitness[best_idx]:
                best_genom = genomes[best_idx]
                best_fitness = fitness[best_idx]
                patience_counter = 0
            else:
                patience_counter += 1
            
            fitness_history.append(best_fitness)

            # select parents
            parents = selection_fn(fitness)

            # crossover
            genomes = crossover_fn(genomes, parents)

            # mutation
            genomes = mutation_fn(genomes)

            print(f"Iteration [{i+1}/{max_iter}]    Best fitness: {best_fitness:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return best_genom, fitness_history