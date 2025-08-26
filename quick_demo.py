"""
TSP Genetic Algorithm - Quick Demonstration
Lab 10: Artificial Intelligence

This script provides a quick demonstration of the key features
"""

from tsp_genetic_algorithm import TSPGeneticAlgorithm
from simple_tsp_demo import SimpleTSPGA, create_example_cities
from berlin52_data import get_berlin52_subset
import matplotlib.pyplot as plt

def demo_basic_concepts():
    """Demonstrate basic GA concepts with small example"""
    print("=" * 50)
    print("DEMO 1: Basic GA Concepts")
    print("=" * 50)
    
    cities = create_example_cities()[:10]  # Use 10 cities for quick demo
    print(f"Testing with {len(cities)} cities")
    
    # Show initial random solution vs GA solution
    import random
    random_route = list(range(len(cities)))
    random.shuffle(random_route)
    
    # Calculate random route distance
    def calc_distance(route, cities):
        total = 0
        for i in range(len(route)):
            x1, y1 = cities[route[i]]
            x2, y2 = cities[route[(i + 1) % len(route)]]
            total += ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        return total
    
    random_distance = calc_distance(random_route, cities)
    print(f"Random solution distance: {random_distance:.2f}")
    
    # Run GA
    ga = SimpleTSPGA(cities, pop_size=30, generations=50, mutation_rate=0.1)
    best_route, best_distance = ga.run()
    
    improvement = ((random_distance - best_distance) / random_distance) * 100
    print(f"GA solution distance: {best_distance:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    
    ga.plot_solution(best_route)

def demo_algorithm_components():
    """Demonstrate different GA components"""
    print("=" * 50)
    print("DEMO 2: Algorithm Components")
    print("=" * 50)
    
    cities = get_berlin52_subset(12)
    print(f"Testing components with {len(cities)} cities")
    
    components = [
        ("Tournament + Order + Swap", "tournament", "order", "swap"),
        ("Roulette + Cycle + Inversion", "roulette", "cycle", "inversion")
    ]
    
    for name, selection, crossover, mutation in components:
        print(f"\nTesting: {name}")
        print("-" * 30)
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=50,
            mutation_rate=0.1,
            elite_size=10,
            generations=100
        )
        
        best_route, best_distance = ga.run_algorithm(
            selection_method=selection,
            crossover_method=crossover,
            mutation_method=mutation,
            verbose=False
        )
        
        print(f"Result: {best_distance:.2f}")

def demo_convergence():
    """Demonstrate convergence behavior"""
    print("=" * 50)
    print("DEMO 3: Convergence Analysis")
    print("=" * 50)
    
    cities = get_berlin52_subset(10)
    
    ga = TSPGeneticAlgorithm(
        cities=cities,
        population_size=60,
        mutation_rate=0.1,
        elite_size=12,
        generations=200
    )
    
    best_route, best_distance = ga.run_algorithm(verbose=True)
    
    # Plot convergence
    import os
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(ga.best_distances, label='Best Distance', linewidth=2)
    plt.plot(ga.average_distances, label='Average Distance', alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Show improvement rate
    improvements = []
    for i in range(1, len(ga.best_distances)):
        if ga.best_distances[i] < ga.best_distances[i-1]:
            improvements.append(i)
    
    plt.scatter(improvements, [ga.best_distances[i] for i in improvements], 
                c='red', s=50, alpha=0.7)
    plt.plot(ga.best_distances, 'b-', alpha=0.5)
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.title(f'Improvements (Total: {len(improvements)})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save convergence analysis
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/convergence_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")
    
    plt.show()
    
    print(f"\nFinal Results:")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Number of improvements: {len(improvements)}")
    print(f"Convergence rate: {len(improvements)/len(ga.best_distances)*100:.1f}%")

def main():
    """Run all demonstrations"""
    print("TSP GENETIC ALGORITHM - QUICK DEMONSTRATION")
    print("Lab 10: Artificial Intelligence")
    print("=" * 50)
    
    try:
        print("\n1. Basic GA Concepts")
        demo_basic_concepts()
        
        print("\n2. Algorithm Components")
        demo_algorithm_components()
        
        print("\n3. Convergence Analysis")
        demo_convergence()
        
        print("\n" + "=" * 50)
        print("DEMONSTRATION COMPLETED!")
        print("=" * 50)
        print("\nKey Takeaways:")
        print("✓ GA significantly improves over random solutions")
        print("✓ Different components affect performance")
        print("✓ Algorithm shows clear convergence behavior")
        print("✓ Early stopping prevents unnecessary computation")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    main()
