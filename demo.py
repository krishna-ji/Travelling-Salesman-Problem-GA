"""
Simple TSP Genetic Algorithm Demo
Lab 10: Artificial Intelligence

This is a simplified version focusing on core GA concepts for TSP
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

class SimpleTSPGA:
    def __init__(self, cities, pop_size=50, generations=200, mutation_rate=0.1):
        self.cities = cities
        self.num_cities = len(cities)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Track best fitness over generations
        self.best_fitness_history = []
    
    def distance(self, city1, city2):
        """Calculate Euclidean distance between two cities"""
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def route_distance(self, route):
        """Calculate total distance of a route"""
        total = 0
        for i in range(len(route)):
            total += self.distance(route[i], route[(i + 1) % len(route)])
        return total
    
    def fitness(self, route):
        """Fitness function - higher is better (inverse of distance)"""
        return 1 / self.route_distance(route)
    
    def create_population(self):
        """Create initial random population"""
        population = []
        for _ in range(self.pop_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(route)
        return population
    
    def tournament_selection(self, population, k=3):
        """Select parent using tournament selection"""
        tournament = random.sample(population, k)
        return max(tournament, key=self.fitness)
    
    def order_crossover(self, parent1, parent2):
        """Order crossover - preserves city order"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Child inherits a slice from parent1
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        remaining = [city for city in parent2 if city not in child]
        
        j = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[j]
                j += 1
        
        return child
    
    def mutate(self, route):
        """Swap mutation - swap two random cities"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def run(self):
        """Run the genetic algorithm"""
        print(f"Running GA for TSP with {self.num_cities} cities")
        print(f"Population: {self.pop_size}, Generations: {self.generations}")
        print("-" * 50)
        
        # Initialize population
        population = self.create_population()
        
        best_ever = None
        best_distance = float('inf')
        
        for gen in range(self.generations):
            # Evaluate population
            fitness_scores = [self.fitness(route) for route in population]
            
            # Track best solution
            best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            current_best = population[best_idx]
            current_distance = self.route_distance(current_best)
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_ever = current_best.copy()
            
            self.best_fitness_history.append(1/best_distance)
            
            # Print progress
            if gen % 50 == 0:
                avg_distance = sum(self.route_distance(route) for route in population) / len(population)
                print(f"Gen {gen:3d}: Best={best_distance:.2f}, Avg={avg_distance:.2f}")
            
            # Create next generation
            new_population = []
            
            # Keep best individual (elitism)
            new_population.append(best_ever.copy())
            
            # Generate rest through crossover and mutation
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child = self.order_crossover(parent1, parent2)
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        print(f"\nFinal best distance: {best_distance:.2f}")
        return best_ever, best_distance
    
    def plot_solution(self, best_route, save_to_file=True):
        """Plot the best route found"""
        import os
        plt.figure(figsize=(12, 5))
        
        # Plot fitness evolution
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history)
        plt.title('Fitness Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (1/Distance)')
        plt.grid(True)
        
        # Plot best route
        plt.subplot(1, 2, 2)
        
        # Plot cities
        x_coords = [self.cities[i][0] for i in range(self.num_cities)]
        y_coords = [self.cities[i][1] for i in range(self.num_cities)]
        plt.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
        
        # Plot route
        route_x = [self.cities[i][0] for i in best_route] + [self.cities[best_route[0]][0]]
        route_y = [self.cities[i][1] for i in best_route] + [self.cities[best_route[0]][1]]
        plt.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.7)
        
        # Label cities
        for i in range(self.num_cities):
            plt.annotate(str(i), (self.cities[i][0], self.cities[i][1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Best Route (Distance: {self.route_distance(best_route):.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save to file for lab report
        if save_to_file:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/simple_tsp_demo_result.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {filename}")
        
        plt.show()


# Example cities
def create_example_cities():
    """Create a small set of example cities"""
    return [
        (60, 200), (180, 200), (80, 180), (140, 180),
        (20, 160), (100, 160), (200, 160), (140, 140),
        (40, 120), (100, 120), (180, 100), (60, 80),
        (120, 80), (180, 60), (20, 40), (100, 40),
        (200, 40), (20, 20), (60, 20), (160, 20)
    ]

def demo():
    """Run a simple demonstration"""
    print("TSP Genetic Algorithm Demo")
    print("=" * 30)
    
    # Create cities
    cities = create_example_cities()
    
    # Run GA
    ga = SimpleTSPGA(cities, pop_size=100, generations=300, mutation_rate=0.1)
    best_route, best_distance = ga.run()
    
    print(f"\nBest route found: {best_route}")
    print(f"Route visits cities in order: {' -> '.join(map(str, best_route))} -> {best_route[0]}")
    
    # Plot results
    ga.plot_solution(best_route)

if __name__ == "__main__":
    demo()
