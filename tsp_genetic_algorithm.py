"""
Traveling Salesman Problem (TSP) using Genetic Algorithm
Lab 10: Artificial Intelligence

This implementation solves TSP using Genetic Algorithm with:
- Tournament and Roulette-wheel selection
- Order crossover and cycle crossover
- Swap and inversion mutation
- Elitism for replacement strategy
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple, Dict
import time

class TSPGeneticAlgorithm:
    def __init__(self, cities: List[Tuple[float, float]], population_size: int = 100, 
                 mutation_rate: float = 0.1, elite_size: int = 20, generations: int = 500):
        """
        Initialize TSP Genetic Algorithm
        
        Args:
            cities: List of (x, y) coordinates for each city
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation (0.1 = 10%)
            elite_size: Number of best individuals to keep each generation
            generations: Maximum number of generations to run
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations
        
        # Calculate distance matrix for efficiency
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Track evolution progress
        self.best_distances = []
        self.average_distances = []
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix between all cities"""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return matrix
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a given route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Return to start
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
    
    def fitness_function(self, route: List[int]) -> float:
        """
        Fitness function - higher is better
        Using reciprocal of distance so shorter routes have higher fitness
        """
        distance = self.calculate_route_distance(route)
        return 1 / distance if distance > 0 else float('inf')
    
    def create_initial_population(self) -> List[List[int]]:
        """Generate initial population of random permutations"""
        population = []
        base_route = list(range(self.num_cities))
        
        for _ in range(self.population_size):
            route = base_route.copy()
            random.shuffle(route)
            population.append(route)
        
        return population
    
    def tournament_selection(self, population: List[List[int]], tournament_size: int = 5) -> List[int]:
        """Tournament selection - pick k random individuals, return the best"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.fitness_function)
    
    def roulette_wheel_selection(self, population: List[List[int]]) -> List[int]:
        """Roulette wheel selection - probabilistic based on fitness"""
        fitness_scores = [self.fitness_function(individual) for individual in population]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            return random.choice(population)
        
        # Normalize fitness scores to probabilities
        probabilities = [f / total_fitness for f in fitness_scores]
        
        # Roulette wheel spin
        spin = random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if spin <= cumulative_prob:
                return population[i]
        
        return population[-1]  # Fallback
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) - maintains relative order from parents
        """
        size = len(parent1)
        
        # Select random slice from parent1
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        
        # Child 1: Keep slice from parent1, fill rest from parent2 in order
        child1 = [-1] * size
        child1[start:end] = parent1[start:end]
        
        # Fill remaining positions with cities from parent2 in order
        parent2_filtered = [city for city in parent2 if city not in child1]
        fill_positions = [i for i in range(size) if child1[i] == -1]
        
        for i, pos in enumerate(fill_positions):
            child1[pos] = parent2_filtered[i]
        
        # Child 2: Keep slice from parent2, fill rest from parent1 in order
        child2 = [-1] * size
        child2[start:end] = parent2[start:end]
        
        parent1_filtered = [city for city in parent1 if city not in child2]
        fill_positions = [i for i in range(size) if child2[i] == -1]
        
        for i, pos in enumerate(fill_positions):
            child2[pos] = parent1_filtered[i]
        
        return child1, child2
    
    def cycle_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Cycle Crossover (CX) - preserves absolute positions
        """
        size = len(parent1)
        child1 = [-1] * size
        child2 = [-1] * size
        
        visited = [False] * size
        
        for start in range(size):
            if visited[start]:
                continue
                
            # Find cycle starting from this position
            cycle_indices = []
            current = start
            
            while not visited[current]:
                visited[current] = True
                cycle_indices.append(current)
                # Find where parent1[current] appears in parent2
                current = parent2.index(parent1[current])
            
            # Alternate which parent contributes to which child for each cycle
            if len([i for i in range(start) if not visited[i]]) % 2 == 0:
                for idx in cycle_indices:
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
            else:
                for idx in cycle_indices:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]
        
        return child1, child2
    
    def swap_mutation(self, route: List[int]) -> List[int]:
        """Swap mutation - randomly swap two cities"""
        mutated = route.copy()
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def inversion_mutation(self, route: List[int]) -> List[int]:
        """Inversion mutation - reverse a random segment"""
        mutated = route.copy()
        if random.random() < self.mutation_rate:
            start = random.randint(0, len(route) - 2)
            end = random.randint(start + 1, len(route))
            mutated[start:end] = reversed(mutated[start:end])
        return mutated
    
    def select_parents(self, population: List[List[int]], method: str = 'tournament') -> Tuple[List[int], List[int]]:
        """Select two parents using specified method"""
        if method == 'tournament':
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
        elif method == 'roulette':
            parent1 = self.roulette_wheel_selection(population)
            parent2 = self.roulette_wheel_selection(population)
        else:
            raise ValueError("Selection method must be 'tournament' or 'roulette'")
        
        return parent1, parent2
    
    def evolve_population(self, population: List[List[int]], 
                         selection_method: str = 'tournament',
                         crossover_method: str = 'order',
                         mutation_method: str = 'swap') -> List[List[int]]:
        """Evolve population for one generation"""
        
        # Sort population by fitness (best first)
        population.sort(key=self.fitness_function, reverse=True)
        
        # Keep elite individuals
        new_population = population[:self.elite_size].copy()
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1, parent2 = self.select_parents(population, selection_method)
            
            # Crossover
            if crossover_method == 'order':
                child1, child2 = self.order_crossover(parent1, parent2)
            elif crossover_method == 'cycle':
                child1, child2 = self.cycle_crossover(parent1, parent2)
            else:
                raise ValueError("Crossover method must be 'order' or 'cycle'")
            
            # Mutation
            if mutation_method == 'swap':
                child1 = self.swap_mutation(child1)
                child2 = self.swap_mutation(child2)
            elif mutation_method == 'inversion':
                child1 = self.inversion_mutation(child1)
                child2 = self.inversion_mutation(child2)
            else:
                raise ValueError("Mutation method must be 'swap' or 'inversion'")
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def run_algorithm(self, selection_method: str = 'tournament',
                     crossover_method: str = 'order',
                     mutation_method: str = 'swap',
                     verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run the genetic algorithm
        
        Returns:
            best_route: Best route found
            best_distance: Distance of best route
        """
        print(f"Starting TSP Genetic Algorithm with {self.num_cities} cities")
        print(f"Population size: {self.population_size}, Generations: {self.generations}")
        print(f"Selection: {selection_method}, Crossover: {crossover_method}, Mutation: {mutation_method}")
        print("-" * 60)
        
        # Initialize population
        population = self.create_initial_population()
        
        best_route = None
        best_distance = float('inf')
        no_improvement_count = 0
        
        start_time = time.time()
        
        for generation in range(self.generations):
            # Evolve population
            population = self.evolve_population(population, selection_method, 
                                              crossover_method, mutation_method)
            
            # Track statistics
            distances = [self.calculate_route_distance(route) for route in population]
            current_best_distance = min(distances)
            avg_distance = sum(distances) / len(distances)
            
            self.best_distances.append(current_best_distance)
            self.average_distances.append(avg_distance)
            
            # Update best solution
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = population[distances.index(current_best_distance)].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Progress reporting
            if verbose and (generation % 50 == 0 or generation == self.generations - 1):
                elapsed_time = time.time() - start_time
                print(f"Generation {generation:4d}: Best = {best_distance:.2f}, "
                      f"Avg = {avg_distance:.2f}, Time = {elapsed_time:.1f}s")
            
            # Early stopping if no improvement for 100 generations
            if no_improvement_count >= 100:
                if verbose:
                    print(f"Early stopping at generation {generation} (no improvement for 100 generations)")
                break
        
        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.2f} seconds")
        print(f"Best route distance: {best_distance:.2f}")
        
        return best_route, best_distance
    
    def plot_results(self, best_route: List[int], title: str = "TSP Solution", save_to_file: bool = True):
        """Plot the evolution progress and best route"""
        import os
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot evolution progress
        ax1.plot(self.best_distances, label='Best Distance', linewidth=2)
        ax1.plot(self.average_distances, label='Average Distance', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Distance')
        ax1.set_title('Evolution Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot best route
        route_cities = [self.cities[i] for i in best_route]
        route_cities.append(route_cities[0])  # Return to start
        
        x_coords = [city[0] for city in route_cities]
        y_coords = [city[1] for city in route_cities]
        
        ax2.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
        ax2.scatter([city[0] for city in self.cities], 
                   [city[1] for city in self.cities], 
                   c='red', s=50, zorder=5)
        
        # Label cities
        for i, city in enumerate(self.cities):
            ax2.annotate(str(i), (city[0], city[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title(f'{title}\nDistance: {self.calculate_route_distance(best_route):.2f}')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save to file for lab report
        if save_to_file:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            # Clean title for filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            filename = f"{output_dir}/{safe_title}_result.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {filename}")
        
        plt.show()


def load_berlin52_dataset() -> List[Tuple[float, float]]:
    """
    Load Berlin52 dataset coordinates
    Returns first 20 cities for demonstration (you can use all 52)
    """
    # Berlin52 dataset coordinates (subset for demonstration)
    berlin52_coords = [
        (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0), (845.0, 655.0),
        (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0), (580.0, 1175.0), (650.0, 1130.0),
        (1605.0, 620.0), (1220.0, 580.0), (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0),
        (725.0, 370.0), (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
        (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0), (975.0, 580.0),
        (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0), (660.0, 180.0), (410.0, 250.0),
        (420.0, 555.0), (575.0, 665.0), (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0),
        (685.0, 610.0), (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
        (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0), (555.0, 815.0),
        (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0), (605.0, 625.0), (595.0, 360.0),
        (1340.0, 725.0), (1740.0, 245.0)
    ]
    
    # Return first 20 cities for faster computation (you can return all 52)
    return berlin52_coords[:20]


def create_random_cities(num_cities: int = 10, grid_size: int = 1000) -> List[Tuple[float, float]]:
    """Create random cities for testing"""
    cities = []
    for _ in range(num_cities):
        x = random.uniform(0, grid_size)
        y = random.uniform(0, grid_size)
        cities.append((x, y))
    return cities


def compare_algorithms():
    """Compare different GA configurations"""
    print("=" * 70)
    print("TSP GENETIC ALGORITHM COMPARISON")
    print("=" * 70)
    
    # Load cities
    cities = load_berlin52_dataset()
    print(f"Testing with {len(cities)} cities from Berlin52 dataset")
    
    configurations = [
        ('Tournament + Order + Swap', 'tournament', 'order', 'swap'),
        ('Tournament + Order + Inversion', 'tournament', 'order', 'inversion'),
        ('Roulette + Order + Swap', 'roulette', 'order', 'swap'),
        ('Tournament + Cycle + Swap', 'tournament', 'cycle', 'swap'),
    ]
    
    results = []
    
    for name, selection, crossover, mutation in configurations:
        print(f"\n{'-' * 50}")
        print(f"Testing: {name}")
        print(f"{'-' * 50}")
        
        # Create GA instance
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=100,
            mutation_rate=0.1,
            elite_size=20,
            generations=300
        )
        
        # Run algorithm
        best_route, best_distance = ga.run_algorithm(
            selection_method=selection,
            crossover_method=crossover,
            mutation_method=mutation,
            verbose=False
        )
        
        results.append((name, best_distance, best_route))
        
        # Plot results
        ga.plot_results(best_route, f"{name}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Configuration':<35} {'Best Distance':<15}")
    print(f"{'-' * 50}")
    
    results.sort(key=lambda x: x[1])  # Sort by distance
    for name, distance, _ in results:
        print(f"{name:<35} {distance:<15.2f}")


if __name__ == "__main__":
    # Example usage
    print("TSP Genetic Algorithm Implementation")
    print("Choose option:")
    print("1. Run single algorithm")
    print("2. Compare multiple configurations")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single run example
        cities = load_berlin52_dataset()
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=100,
            mutation_rate=0.1,
            elite_size=20,
            generations=500
        )
        
        best_route, best_distance = ga.run_algorithm(
            selection_method='tournament',
            crossover_method='order',
            mutation_method='swap'
        )
        
        print(f"\nBest route: {best_route}")
        print(f"Best distance: {best_distance:.2f}")
        
        ga.plot_results(best_route, "TSP Solution - Tournament + Order + Swap")
        
    elif choice == "2":
        # Comparison mode
        compare_algorithms()
    
    else:
        print("Invalid choice!")

