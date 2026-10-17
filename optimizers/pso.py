import torch

class ParticleSwarmOptimizer:
    def __init__(self, population, dim, fitness_fn, device):
        self.population = population
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.device = device

    def train(self, w=0.7, cp=1.5, cg=1.5, max_iter=50, patience=10):
        positions = torch.rand((self.population, self.dim), device=self.device) * 2 - 1
        velocities = torch.rand((self.population, self.dim), device=self.device) * 4 - 2

        best_positions = positions.clone()
        best_values = self.fitness_fn(positions).to(self.device)

        best_idx = torch.argmax(best_values)
        global_best_position = best_positions[best_idx.item()]
        global_best_value = best_values[best_idx.item()]

        fitness_history = [global_best_value]

        patience_counter = 0

        for i in range(max_iter):
            # Update particles
            rp = torch.rand((self.population, self.dim), device=self.device)
            rg = torch.rand((self.population, self.dim), device=self.device)

            velocities = w * velocities + cp * rp * (best_positions - positions) + cg * rg * (global_best_position - positions)
            positions = positions + velocities

            # Calculate fitness
            fitness = self.fitness_fn(positions).to(self.device)
            
            # Update best records
            mask = fitness > best_values
            best_positions[mask, :] = positions[mask, :]
            best_values[mask] = fitness[mask]

            if mask.any():
                best_idx = torch.argmax(best_values[mask])
                if global_best_value < best_values[mask][best_idx.item()]:
                    global_best_position = best_positions[mask][best_idx.item()]
                    global_best_value = best_values[mask][best_idx.item()]
                    patience_counter = 0
                else:
                    patience_counter += 1

            fitness_history.append(global_best_value)

            print(f"Iteration [{i+1}/{max_iter}]    Best fitness: {global_best_value:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return global_best_position, fitness_history
