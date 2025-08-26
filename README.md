# TSP using Genetic Algorithm - Lab 10

This project implements a complete Genetic Algorithm solution for the Traveling Salesman Problem (TSP) as part of Lab 10 for Artificial Intelligence course.

## Project Structure

```
tsp-using-ga/
├── tsp_genetic_algorithm.py    # Complete GA implementation with all features
├── simple_tsp_demo.py         # Simplified demo focusing on core concepts
├── berlin52_data.py           # Berlin52 dataset (standard TSP benchmark)
├── test_suite.py              # Comprehensive testing framework
└── README.md                  # This file
```

## Features Implemented

### 1. Problem Setup ✅
- **Cities representation**: Coordinates (x, y) for each city
- **Distance calculation**: Euclidean distance between cities
- **Berlin52 dataset**: Standard TSP benchmark with 52 cities

### 2. Solution Representation ✅
- **Chromosome**: Permutation of city indices
- **Example**: [2, 0, 3, 1] means visiting cities in order 2→0→3→1→2

### 3. Fitness Function ✅
- **Objective**: Minimize total route distance
- **Implementation**: Fitness = 1/distance (higher fitness = better solution)

### 4. Population Initialization ✅
- **Method**: Random permutations of city indices
- **Constraint**: Each route visits all cities exactly once

### 5. Selection Methods ✅
- **Tournament Selection**: Pick k random solutions, select the best
- **Roulette Wheel Selection**: Probabilistic selection based on fitness

### 6. Crossover Operations ✅
- **Order Crossover (OX)**: Preserves relative city order from parents
- **Cycle Crossover (CX)**: Preserves absolute city positions

### 7. Mutation Operations ✅
- **Swap Mutation**: Randomly swap two cities
- **Inversion Mutation**: Reverse a random segment of the route
- **Mutation Rate**: Configurable (default 10%)

### 8. Replacement Strategy ✅
- **Elitism**: Keep best individuals across generations
- **Population replacement**: Combine elites with new offspring

### 9. Stopping Criteria ✅
- **Fixed generations**: Run for specified number of generations
- **Early stopping**: Stop if no improvement for 100 generations
- **Convergence tracking**: Monitor best and average fitness

### 10. Visualization ✅
- **Evolution progress**: Plot fitness over generations
- **Best route visualization**: Display optimal path found
- **City labeling**: Show city indices and connections

## Quick Start

### 1. Simple Demo
```python
python simple_tsp_demo.py
```
This runs a basic demonstration with example cities.

### 2. Complete Implementation
```python
python tsp_genetic_algorithm.py
```
Choose between single run or algorithm comparison.

### 3. Comprehensive Testing
```python
python test_suite.py
```
Run various tests to compare different GA configurations.

## Usage Examples

### Basic Usage
```python
from tsp_genetic_algorithm import TSPGeneticAlgorithm
from berlin52_data import get_berlin52_subset

# Load cities
cities = get_berlin52_subset(20)  # Use 20 cities from Berlin52

# Create GA instance
ga = TSPGeneticAlgorithm(
    cities=cities,
    population_size=100,
    mutation_rate=0.1,
    elite_size=20,
    generations=500
)

# Run algorithm
best_route, best_distance = ga.run_algorithm(
    selection_method='tournament',
    crossover_method='order',
    mutation_method='swap'
)

# Visualize results
ga.plot_results(best_route, "TSP Solution")
```

### Configuration Options

#### Selection Methods
- `'tournament'`: Tournament selection (recommended)
- `'roulette'`: Roulette wheel selection

#### Crossover Methods
- `'order'`: Order crossover (recommended for TSP)
- `'cycle'`: Cycle crossover

#### Mutation Methods
- `'swap'`: Swap two random cities
- `'inversion'`: Reverse a route segment

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Number of individuals in population |
| `mutation_rate` | 0.1 | Probability of mutation (10%) |
| `elite_size` | 20 | Number of best individuals to preserve |
| `generations` | 500 | Maximum generations to run |

## Performance Results

### Berlin52 Subset (20 cities)
- **Typical results**: 2000-3000 distance units
- **Computation time**: 10-30 seconds
- **Convergence**: Usually within 200-300 generations

### Algorithm Comparison
Based on testing with Berlin52 subset:

1. **Tournament + Order + Swap**: Generally best performance
2. **Tournament + Order + Inversion**: Good exploration
3. **Roulette + Order + Swap**: Decent but slower convergence
4. **Tournament + Cycle + Swap**: Variable performance

## Key Concepts Demonstrated

### 1. Genetic Algorithm Components
- **Representation**: Permutation encoding for TSP
- **Population**: Multiple candidate solutions
- **Evolution**: Iterative improvement through generations

### 2. Selection Pressure
- **Tournament**: Controlled selection pressure
- **Roulette Wheel**: Fitness-proportionate selection

### 3. Genetic Operators
- **Crossover**: Combine good solutions
- **Mutation**: Introduce diversity
- **Elitism**: Preserve best solutions

### 4. Problem-Specific Considerations
- **Permutation constraints**: Valid TSP routes
- **Distance calculation**: Euclidean geometry
- **Optimization objective**: Minimize tour length

## Educational Value

This implementation covers all major GA concepts:

1. **Problem encoding**: How to represent TSP as GA problem
2. **Fitness landscape**: Understanding TSP optimization
3. **Operator design**: TSP-specific crossover and mutation
4. **Parameter tuning**: Effect of population size, mutation rate
5. **Performance analysis**: Convergence and solution quality

## Extensions and Improvements

### Possible Enhancements
1. **Advanced operators**: PMX crossover, 2-opt mutation
2. **Hybrid approaches**: Local search integration
3. **Adaptive parameters**: Dynamic mutation rates
4. **Parallel processing**: Multi-threaded evolution
5. **Larger datasets**: Full Berlin52 or other benchmarks

### Real-world Applications
- **Route optimization**: Delivery, logistics
- **Manufacturing**: PCB drilling, robot path planning
- **Bioinformatics**: DNA sequencing
- **Network design**: Minimum spanning trees

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

## Running the Code

1. **Clone or download** the project files
2. **Install requirements**: `pip install numpy matplotlib`
3. **Run demo**: `python simple_tsp_demo.py`
4. **Run tests**: `python test_suite.py`
5. **Experiment**: Modify parameters and observe results

## Lab Report Guidelines

When writing your lab report, consider:

1. **Algorithm description**: Explain each GA component
2. **Implementation details**: Key design decisions
3. **Experimental setup**: Parameters and test cases
4. **Results analysis**: Compare different configurations
5. **Performance evaluation**: Quality vs. computation time
6. **Conclusions**: What works best and why

## Author

Created for AI Lab 10 - TSP using Genetic Algorithm

## License

Educational use only - part of AI course lab assignment.
