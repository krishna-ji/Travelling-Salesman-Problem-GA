"""
Generate Individual Figures for Lab Report
TSP Genetic Algorithm - Lab 10
One curve/analysis per plot for cleaner presentation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tsp_genetic_algorithm import TSPGeneticAlgorithm, create_random_cities
from demo import SimpleTSPGA, create_example_cities
from berlin52_data import get_berlin52_subset, get_berlin52_data
import time

# Set matplotlib to not show plots interactively (save only)
plt.ioff()

def ensure_output_dir():
    """Ensure output directory exists"""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_figure(filename, dpi=300):
    """Save current figure with high quality"""
    output_dir = ensure_output_dir()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def fig1_random_vs_ga():
    """Figure 1: Random Solution vs GA Solution Comparison"""
    print("Generating Figure 1: Random vs GA Comparison...")
    
    cities = create_example_cities()[:12]
    
    # Random solution
    import random
    random.seed(42)
    random_route = list(range(len(cities)))
    random.shuffle(random_route)
    
    def calc_distance(route, cities):
        total = 0
        for i in range(len(route)):
            x1, y1 = cities[route[i]]
            x2, y2 = cities[route[(i + 1) % len(route)]]
            total += ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        return total
    
    random_distance = calc_distance(random_route, cities)
    
    # GA solution
    ga = SimpleTSPGA(cities, pop_size=50, generations=100, mutation_rate=0.1)
    best_route, best_distance = ga.run()
    
    improvement = ((random_distance - best_distance) / random_distance) * 100
    
    plt.figure(figsize=(10, 8))
    
    # Random solution route
    route_x = [cities[i][0] for i in random_route] + [cities[random_route[0]][0]]
    route_y = [cities[i][1] for i in random_route] + [cities[random_route[0]][1]]
    
    plt.plot(route_x, route_y, 'r-', linewidth=3, alpha=0.7, label=f'Random Solution (Distance: {random_distance:.1f})')
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], 
                c='blue', s=150, zorder=5, edgecolors='black', linewidth=2)
    
    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), xytext=(0, 0), 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')
    
    plt.title(f'Random Solution vs Genetic Algorithm\nImprovement: {improvement:.1f}%', 
              fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure("figure_1_random_solution.png")

def fig2_ga_solution():
    """Figure 2: GA Optimized Solution"""
    print("Generating Figure 2: GA Solution...")
    
    cities = create_example_cities()[:12]
    
    # GA solution
    ga = SimpleTSPGA(cities, pop_size=50, generations=100, mutation_rate=0.1)
    best_route, best_distance = ga.run()
    
    plt.figure(figsize=(10, 8))
    
    # GA solution route
    route_x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    route_y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    
    plt.plot(route_x, route_y, 'g-', linewidth=3, alpha=0.8, label=f'GA Solution (Distance: {best_distance:.1f})')
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], 
                c='red', s=150, zorder=5, edgecolors='black', linewidth=2)
    
    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), xytext=(0, 0), 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')
    
    plt.title('Genetic Algorithm Optimized Route', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure("figure_2_ga_solution.png")

def fig3_fitness_evolution():
    """Figure 3: Fitness Evolution Over Generations"""
    print("Generating Figure 3: Fitness Evolution...")
    
    cities = get_berlin52_subset(15)
    
    ga = TSPGeneticAlgorithm(
        cities=cities,
        population_size=100,
        mutation_rate=0.1,
        elite_size=20,
        generations=200
    )
    
    best_route, best_distance = ga.run_algorithm(verbose=False)
    
    plt.figure(figsize=(12, 8))
    plt.plot(ga.best_distances, 'b-', linewidth=3, label='Best Distance')
    plt.title('Genetic Algorithm Convergence', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Best Distance Found', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    save_figure("figure_3_fitness_evolution.png")

def fig4_average_fitness():
    """Figure 4: Average Population Fitness"""
    print("Generating Figure 4: Average Fitness...")
    
    cities = get_berlin52_subset(15)
    
    ga = TSPGeneticAlgorithm(
        cities=cities,
        population_size=100,
        mutation_rate=0.1,
        elite_size=20,
        generations=200
    )
    
    best_route, best_distance = ga.run_algorithm(verbose=False)
    
    plt.figure(figsize=(12, 8))
    plt.plot(ga.average_distances, 'orange', linewidth=3, label='Average Distance')
    plt.title('Average Population Fitness Evolution', fontsize=16, fontweight='bold')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Average Distance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    save_figure("figure_4_average_fitness.png")

def fig5_selection_comparison():
    """Figure 5: Selection Methods Comparison"""
    print("Generating Figure 5: Selection Methods...")
    
    cities = get_berlin52_subset(12)
    
    methods = ['Tournament', 'Roulette']
    distances = []
    times = []
    
    for method in ['tournament', 'roulette']:
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=0.1,
            elite_size=15,
            generations=150
        )
        
        start_time = time.time()
        best_route, best_distance = ga.run_algorithm(
            selection_method=method,
            verbose=False
        )
        end_time = time.time()
        
        distances.append(best_distance)
        times.append(end_time - start_time)
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(methods, distances, color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Selection Method Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Best Distance Found', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        plt.annotate(f'{distance:.0f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure("figure_5_selection_comparison.png")

def fig6_crossover_comparison():
    """Figure 6: Crossover Methods Comparison"""
    print("Generating Figure 6: Crossover Methods...")
    
    cities = get_berlin52_subset(12)
    
    methods = ['Order Crossover', 'Cycle Crossover']
    distances = []
    
    for method in ['order', 'cycle']:
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=0.1,
            elite_size=15,
            generations=150
        )
        
        best_route, best_distance = ga.run_algorithm(
            crossover_method=method,
            verbose=False
        )
        
        distances.append(best_distance)
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(methods, distances, color=['lightgreen', 'gold'], alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Crossover Method Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Best Distance Found', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        plt.annotate(f'{distance:.0f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure("figure_6_crossover_comparison.png")

def fig7_mutation_comparison():
    """Figure 7: Mutation Methods Comparison"""
    print("Generating Figure 7: Mutation Methods...")
    
    cities = get_berlin52_subset(12)
    
    methods = ['Swap Mutation', 'Inversion Mutation']
    distances = []
    
    for method in ['swap', 'inversion']:
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=0.1,
            elite_size=15,
            generations=150
        )
        
        best_route, best_distance = ga.run_algorithm(
            mutation_method=method,
            verbose=False
        )
        
        distances.append(best_distance)
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(methods, distances, color=['purple', 'cyan'], alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Mutation Method Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Best Distance Found', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        plt.annotate(f'{distance:.0f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure("figure_7_mutation_comparison.png")

def fig8_population_size_effect():
    """Figure 8: Population Size Effect"""
    print("Generating Figure 8: Population Size Effect...")
    
    cities = get_berlin52_subset(10)
    pop_sizes = [30, 50, 80, 120, 150]
    distances = []
    
    for pop_size in pop_sizes:
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=pop_size,
            mutation_rate=0.1,
            elite_size=max(5, pop_size // 10),
            generations=100
        )
        
        best_route, best_distance = ga.run_algorithm(verbose=False)
        distances.append(best_distance)
    
    plt.figure(figsize=(12, 8))
    plt.plot(pop_sizes, distances, 'bo-', linewidth=3, markersize=10)
    plt.title('Effect of Population Size on Solution Quality', fontsize=16, fontweight='bold')
    plt.xlabel('Population Size', fontsize=14)
    plt.ylabel('Best Distance Found', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for x, y in zip(pop_sizes, distances):
        plt.annotate(f'{y:.0f}', (x, y), xytext=(0, 15), 
                    textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure("figure_8_population_size.png")

def fig9_mutation_rate_effect():
    """Figure 9: Mutation Rate Effect"""
    print("Generating Figure 9: Mutation Rate Effect...")
    
    cities = get_berlin52_subset(10)
    mutation_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
    distances = []
    
    for mut_rate in mutation_rates:
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=mut_rate,
            elite_size=15,
            generations=100
        )
        
        best_route, best_distance = ga.run_algorithm(verbose=False)
        distances.append(best_distance)
    
    plt.figure(figsize=(12, 8))
    plt.plot(mutation_rates, distances, 'ro-', linewidth=3, markersize=10)
    plt.title('Effect of Mutation Rate on Solution Quality', fontsize=16, fontweight='bold')
    plt.xlabel('Mutation Rate', fontsize=14)
    plt.ylabel('Best Distance Found', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for x, y in zip(mutation_rates, distances):
        plt.annotate(f'{y:.0f}', (x, y), xytext=(0, 15), 
                    textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure("figure_9_mutation_rate.png")

def fig10_scalability():
    """Figure 10: Algorithm Scalability"""
    print("Generating Figure 10: Scalability Analysis...")
    
    problem_sizes = [8, 12, 16, 20]
    distances = []
    times = []
    
    for size in problem_sizes:
        cities = get_berlin52_subset(size)
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=min(100, size * 5),
            mutation_rate=0.1,
            elite_size=min(20, size),
            generations=150
        )
        
        start_time = time.time()
        best_route, best_distance = ga.run_algorithm(verbose=False)
        end_time = time.time()
        
        distances.append(best_distance)
        times.append(end_time - start_time)
    
    plt.figure(figsize=(12, 8))
    plt.plot(problem_sizes, times, 'go-', linewidth=3, markersize=10)
    plt.title('Algorithm Scalability - Computation Time vs Problem Size', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Cities', fontsize=14)
    plt.ylabel('Computation Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for x, y in zip(problem_sizes, times):
        plt.annotate(f'{y:.2f}s', (x, y), xytext=(0, 15), 
                    textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure("figure_10_scalability.png")

def main():
    """Generate all individual figures"""
    print("🎨 GENERATING INDIVIDUAL FIGURES FOR LAB REPORT")
    print("=" * 50)
    
    output_dir = ensure_output_dir()
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        fig1_random_vs_ga()
        fig2_ga_solution()
        fig3_fitness_evolution()
        fig4_average_fitness()
        fig5_selection_comparison()
        fig6_crossover_comparison()
        fig7_mutation_comparison()
        fig8_population_size_effect()
        fig9_mutation_rate_effect()
        fig10_scalability()
        
        print("\n" + "=" * 50)
        print("🎉 ALL INDIVIDUAL FIGURES GENERATED!")
        print("=" * 50)
        
        # List all generated files
        print("\n📁 Generated figures:")
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith('figure_') and filename.endswith('.png'):
                print(f"  📊 {filename}")
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
