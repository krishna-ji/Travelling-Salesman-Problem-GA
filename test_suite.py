"""
TSP Genetic Algorithm - Comprehensive Test Suite
Lab 10: Artificial Intelligence

This script demonstrates all components of the TSP GA implementation
"""

from tsp_genetic_algorithm import TSPGeneticAlgorithm, create_random_cities
from simple_tsp_demo import SimpleTSPGA, create_example_cities
from berlin52_data import get_berlin52_data, get_berlin52_subset
import matplotlib.pyplot as plt
import time

def test_basic_functionality():
    """Test basic GA functionality with simple cities"""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)
    
    cities = create_example_cities()
    print(f"Testing with {len(cities)} example cities")
    
    ga = SimpleTSPGA(cities, pop_size=50, generations=100, mutation_rate=0.1)
    best_route, best_distance = ga.run()
    
    print(f"Best route: {best_route}")
    print(f"Best distance: {best_distance:.2f}")
    
    ga.plot_solution(best_route)
    print("\nBasic functionality test completed!\n")

def test_selection_methods():
    """Compare tournament vs roulette wheel selection"""
    print("=" * 60)
    print("TEST 2: Selection Methods Comparison")
    print("=" * 60)
    
    cities = get_berlin52_subset(15)  # Use 15 cities for faster computation
    
    results = []
    
    for selection_method in ['tournament', 'roulette']:
        print(f"\nTesting {selection_method} selection...")
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=0.1,
            elite_size=15,
            generations=200
        )
        
        start_time = time.time()
        best_route, best_distance = ga.run_algorithm(
            selection_method=selection_method,
            crossover_method='order',
            mutation_method='swap',
            verbose=False
        )
        end_time = time.time()
        
        results.append((selection_method, best_distance, end_time - start_time))
        
        # Plot results
        ga.plot_results(best_route, f"{selection_method.capitalize()} Selection")
    
    print(f"\n{'-' * 40}")
    print("Selection Method Comparison Results:")
    print(f"{'-' * 40}")
    for method, distance, time_taken in results:
        print(f"{method.capitalize():<12}: Distance = {distance:.2f}, Time = {time_taken:.2f}s")

def test_crossover_methods():
    """Compare order vs cycle crossover"""
    print("=" * 60)
    print("TEST 3: Crossover Methods Comparison")
    print("=" * 60)
    
    cities = get_berlin52_subset(15)
    
    results = []
    
    for crossover_method in ['order', 'cycle']:
        print(f"\nTesting {crossover_method} crossover...")
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=0.1,
            elite_size=15,
            generations=200
        )
        
        start_time = time.time()
        best_route, best_distance = ga.run_algorithm(
            selection_method='tournament',
            crossover_method=crossover_method,
            mutation_method='swap',
            verbose=False
        )
        end_time = time.time()
        
        results.append((crossover_method, best_distance, end_time - start_time))
        
        # Plot results
        ga.plot_results(best_route, f"{crossover_method.capitalize()} Crossover")
    
    print(f"\n{'-' * 40}")
    print("Crossover Method Comparison Results:")
    print(f"{'-' * 40}")
    for method, distance, time_taken in results:
        print(f"{method.capitalize():<8}: Distance = {distance:.2f}, Time = {time_taken:.2f}s")

def test_mutation_methods():
    """Compare swap vs inversion mutation"""
    print("=" * 60)
    print("TEST 4: Mutation Methods Comparison")
    print("=" * 60)
    
    cities = get_berlin52_subset(15)
    
    results = []
    
    for mutation_method in ['swap', 'inversion']:
        print(f"\nTesting {mutation_method} mutation...")
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=0.1,
            elite_size=15,
            generations=200
        )
        
        start_time = time.time()
        best_route, best_distance = ga.run_algorithm(
            selection_method='tournament',
            crossover_method='order',
            mutation_method=mutation_method,
            verbose=False
        )
        end_time = time.time()
        
        results.append((mutation_method, best_distance, end_time - start_time))
        
        # Plot results
        ga.plot_results(best_route, f"{mutation_method.capitalize()} Mutation")
    
    print(f"\n{'-' * 40}")
    print("Mutation Method Comparison Results:")
    print(f"{'-' * 40}")
    for method, distance, time_taken in results:
        print(f"{method.capitalize():<10}: Distance = {distance:.2f}, Time = {time_taken:.2f}s")

def test_parameter_effects():
    """Test different population sizes and mutation rates"""
    print("=" * 60)
    print("TEST 5: Parameter Effects")
    print("=" * 60)
    
    cities = get_berlin52_subset(12)  # Smaller problem for faster testing
    
    # Test population sizes
    print("Testing different population sizes:")
    pop_sizes = [30, 60, 100]
    
    for pop_size in pop_sizes:
        print(f"\nTesting population size: {pop_size}")
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=pop_size,
            mutation_rate=0.1,
            elite_size=max(5, pop_size // 10),
            generations=150
        )
        
        best_route, best_distance = ga.run_algorithm(verbose=False)
        print(f"Population {pop_size}: Best distance = {best_distance:.2f}")
    
    print(f"\n{'-' * 30}")
    
    # Test mutation rates
    print("Testing different mutation rates:")
    mutation_rates = [0.05, 0.1, 0.2]
    
    for mut_rate in mutation_rates:
        print(f"\nTesting mutation rate: {mut_rate}")
        
        ga = TSPGeneticAlgorithm(
            cities=cities,
            population_size=80,
            mutation_rate=mut_rate,
            elite_size=15,
            generations=150
        )
        
        best_route, best_distance = ga.run_algorithm(verbose=False)
        print(f"Mutation rate {mut_rate}: Best distance = {best_distance:.2f}")

def test_berlin52_performance():
    """Test on Berlin52 subset and compare with known optimal"""
    print("=" * 60)
    print("TEST 6: Berlin52 Performance Test")
    print("=" * 60)
    
    # Get Berlin52 data
    full_cities, optimal_tour, optimal_distance = get_berlin52_data()
    cities = get_berlin52_subset(20)  # Use subset for reasonable computation time
    
    print(f"Testing on Berlin52 subset ({len(cities)} cities)")
    print(f"Note: Full Berlin52 optimal distance is {optimal_distance}")
    
    ga = TSPGeneticAlgorithm(
        cities=cities,
        population_size=150,
        mutation_rate=0.1,
        elite_size=30,
        generations=500
    )
    
    best_route, best_distance = ga.run_algorithm(
        selection_method='tournament',
        crossover_method='order',
        mutation_method='swap'
    )
    
    print(f"\nResults for {len(cities)} cities:")
    print(f"Best distance found: {best_distance:.2f}")
    
    # Calculate percentage above optimal (rough estimate)
    # Note: This is not directly comparable since we're using subset
    estimated_optimal_subset = optimal_distance * (len(cities) / 52)
    improvement_potential = ((best_distance - estimated_optimal_subset) / estimated_optimal_subset) * 100
    print(f"Estimated optimal for subset: {estimated_optimal_subset:.2f}")
    print(f"Improvement potential: {improvement_potential:.1f}%")
    
    ga.plot_results(best_route, f"Berlin52 Subset ({len(cities)} cities)")

def run_comprehensive_test():
    """Run all tests"""
    print("TSP GENETIC ALGORITHM - COMPREHENSIVE TEST SUITE")
    print("Lab 10: Artificial Intelligence")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_selection_methods()
        test_crossover_methods()
        test_mutation_methods()
        test_parameter_effects()
        test_berlin52_performance()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Some tests may not have completed.")

if __name__ == "__main__":
    print("TSP Genetic Algorithm Test Suite")
    print("Choose test to run:")
    print("1. Basic functionality test")
    print("2. Selection methods comparison")
    print("3. Crossover methods comparison")
    print("4. Mutation methods comparison")
    print("5. Parameter effects test")
    print("6. Berlin52 performance test")
    print("7. Run all tests")
    
    try:
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            test_basic_functionality()
        elif choice == "2":
            test_selection_methods()
        elif choice == "3":
            test_crossover_methods()
        elif choice == "4":
            test_mutation_methods()
        elif choice == "5":
            test_parameter_effects()
        elif choice == "6":
            test_berlin52_performance()
        elif choice == "7":
            run_comprehensive_test()
        else:
            print("Invalid choice!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
