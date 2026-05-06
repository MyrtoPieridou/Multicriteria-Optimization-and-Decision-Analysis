# Multicriteria-Optimization-and-Decision-Analysis
Multicretiria Optimization and Decision Analysis project

This project focuses on solving a Multi-Objective Optimization problem for bicycle route planning using a Genetic Algorithm (GA). 
The goal is to select $n$ nodes from a set of $m$ available nodes to form a path that balances physical efficiency with rider experience.


# Project Overview: Bicycle Path Optimization
The challenge involves finding a non-repeating path that simultaneously addresses two conflicting goals:
- Minimize Total Distance 
- Maximize Overall Comfort


# Genetic Algorithm Implementation
To explore the search space and avoid local optima, the project implements:

- Initialization: Random generation of valid route sequences.
- Fitness Function: Evaluates paths based on the trade-off between distance and comfort.
- Crossover: Combining parent routes to produce offspring with potentially better traits.
- Mutation: Randomly shuffling nodes within a path to maintain genetic diversity.

# Results & Pareto Front
The algorithm identifies non-dominated solutions (the Pareto Front).
The Trade-off: The results demonstrate that as total distance increases, the total comfort score improves, and vice-versa.
Decision Support: By providing a set of optimal solutions rather than just one, a user can choose a path based on their specific preference for distance vs. comfort.

# Academic Context
Institution: Leiden University
Team members: Yudie Zheng, Myrto Pieridou, Christian Steenis
