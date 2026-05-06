# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:12:58 2024

@author: 57213
"""

import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt


n_total_nodes = 20
selected_nodes = 10
np.random.seed(42)


distance_matrix = np.random.randint(10, 100, size=(n_total_nodes, n_total_nodes))
scenic_matrix = np.random.randint(1, 6, size=(n_total_nodes, n_total_nodes))
smoothness_matrix = np.random.randint(1, 6, size=(n_total_nodes, n_total_nodes))
safety_matrix = np.random.randint(1, 6, size=(n_total_nodes, n_total_nodes))
slope_matrix = np.random.randint(1, 6, size=(n_total_nodes, n_total_nodes))


safety_matrix = 6 - safety_matrix
slope_matrix = 6 - slope_matrix


distance_matrix = (distance_matrix + distance_matrix.T) // 2
scenic_matrix = (scenic_matrix + scenic_matrix.T) // 2
smoothness_matrix = (smoothness_matrix + smoothness_matrix.T) // 2
safety_matrix = (safety_matrix + safety_matrix.T) // 2
slope_matrix = (slope_matrix + slope_matrix.T) // 2
np.fill_diagonal(distance_matrix, 0)
com_matrix = (scenic_matrix + 2 * safety_matrix + slope_matrix + 2 * smoothness_matrix) / 6
np.fill_diagonal(com_matrix, 0)


def comfort_score(route):
    return np.sum([com_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)])


def total_distance(route):
    return np.sum([distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)])


def fitness_function(individual):    
    if len(set(individual)) != len(individual):
        return float('inf'), float('-inf')  

    comfort = comfort_score(individual)
    distance = total_distance(individual)
    
    if comfort <= 20:
        return float('inf'), float('-inf') 

    return distance, -comfort  


creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def generate_individual():
    all_nodes = list(range(n_total_nodes))
    selected = random.sample(all_nodes, selected_nodes)
    return creator.Individual(selected)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def ordered_crossover(ind1, ind2):
    size = len(ind1)
    start, end = sorted(random.sample(range(1, size - 1), 2))
    
    temp1 = ind1[start:end]
    temp2 = ind2[start:end]

    child1 = [None] * size
    child2 = [None] * size

    child1[start:end] = temp1
    child2[start:end] = temp2

    fill_indices = lambda lst: [i for i, x in enumerate(lst) if x is None]

    fill_child = lambda parent, child, fill_index: [
        val for val in parent if val not in child[start:end]
    ]

    child1_indices = fill_indices(child1)
    child2_indices = fill_indices(child2)

    child1_filled = fill_child(ind2, child1, child1_indices)
    child2_filled = fill_child(ind1, child2, child2_indices)

    for idx, val in zip(child1_indices, child1_filled):
        child1[idx] = val
    for idx, val in zip(child2_indices, child2_filled):
        child2[idx] = val

    return creator.Individual(child1), creator.Individual(child2)

def mutate_shuffle(individual, indpb):
    if random.random() < indpb:
        intermediate_nodes = individual[1:-1]
        random.shuffle(intermediate_nodes)
        individual[1:-1] = intermediate_nodes
    return individual,

toolbox.register("mate", ordered_crossover)
toolbox.register("mutate", mutate_shuffle, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", fitness_function)


def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    pareto_front = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)


    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.2, ngen=100, 
        stats=stats, halloffame=hof, verbose=True
    )

    pareto_front.update(pop)

    distances = np.array([d.fitness.values[0] for d in pareto_front])
    comfort_scores = np.array([-d.fitness.values[1] for d in pareto_front])
    
    plt.scatter(distances, comfort_scores, c='red')
    plt.xlabel('Total Distance')
    plt.ylabel('Total Comfort Score')
    plt.title('Pareto Front')
    
    z = np.polyfit(distances, comfort_scores, deg=4)  
    p = np.poly1d(z)


    xp = np.linspace(distances.min(), distances.max(), 100)
    plt.plot(xp, p(xp), '-', color='blue')
    
    plt.show()


    for ind in pareto_front:
        start_node = ind[0]
        end_node = ind[-1]
        route = [start_node] + ind[1:-1] + [end_node]
        print(f"Route: {route}, Total Distance: {total_distance(route)}, Comfort Score: {-ind.fitness.values[1]}")

if __name__ == "__main__":
    main()
