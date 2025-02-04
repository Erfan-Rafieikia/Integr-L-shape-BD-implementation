# Benders Decomposition for The Capacitated Facility Location Problem

This project offers a Python implementation of Benders decomposition to tackle the Capacitated Facility Location Problem (CFLP) using the powerful Gurobi Optimizer.
The code showcases an effective approach to model the CFLP as a Mixed Integer Program (MIP) and efficiently solve it through Benders decomposition, leveraging Gurobi's callback functionality.

Key features of this implementation include:

- A modular and customizable structure that allows easy configuration of the CFLP formulation for various problem parameters and sizes.
- A pedagogical reference for integrating the Gurobi optimizer, formulating advanced MIP models, and applying Benders decomposition.
- An emphasis on Python-based modeling and solution techniques for facility location decisions

## CFLP Overview

The CFLP is a classic optimization problem that involves locating facilities and assigning customers to them, taking into account the capacity limitations of each facility. The objective is to minimize the total cost, which includes both fixed costs for opening facilities and variable costs for serving customers.

### Mathematical Formulation

The CFLP can be formulated as a mixed-integer linear program (MIP) with the following notations:

#### Sets and Parameters:

- $I$: Set of customers.
- $J$: Set of candidate facility locations.
- $f_{j}$: Fixed cost of opening a facility at location $j \in J$.
- $c_{ij}$: Unit transportation cost between facility $j \in J$ and customer $i \in I$.
- $d_{i}$: Demand of customer $i \in I$.
- $u_{j}$: Capacity of facility $j \in J$.

#### Decision Variables:

- $x_{ij}$: Amount of customer $i$'s demand served by facility $j$.
- $y_{j}$: Binary variable indicating whether facility $j$ is opened (1) or closed (0).

#### Objective Function:

The objective is to minimize the total cost, which includes both fixed costs for opening facilities and variable costs for serving customers:

```math
\text{minimize} \quad \sum_{j \in J} f_{j}\, y_{j} + \sum_{i \in I} \sum_{j \in J} c_{ij}\, x_{ij}
```

#### Constraints:

- Demand satisfaction constraint: Each customer's demand must be fully served. 

```math
\sum_{j \in J} x_{ij} \geq d_{i},\ \forall i \in I
```

- Capacity constraint: If a facility is open, the total supply from that facility should not exceed its capacity. If a facility is closed, no supply should be made. 

```math
\sum_{i \in I} x_{ij} \leq u_{j}\, y_{j},\ \forall j \in J
```

- Variable domains:

```math
x \in \mathbb{R}^{I \times J}_+,\ y \in \{0, 1\}^{J}.
```

## Benders Decomposition Approach

This method effectively tackles large-scale CFLPs by decomposing them into smaller, more manageable subproblems. It operates through two main components: the master problem and the subproblem. These problems interact iteratively until the lower bounds converge to the true optimal solution.


### Master Problem

The master problem is responsible for selecting the optimal combination of facility locations to open, while also providing a lower bound on the overall cost. The master problem is a Mixed Integer Program (MIP) that aims to minimize total costs, which are composed of fixed costs for opening facilities and an auxiliary variable, $\eta$. The auxiliary variable $\eta$ serves as an underestimate for the optimal objective function value of the subproblem. Here's the refined formulation:

```math
\begin{align*}
    & \text{minimize} \  && \sum_{j \in J} f_{j}\, y_{j} + \eta \\
    & \text{subject to} \  && \sum_{j \in J} u_{j}\, y_{j} \geq \sum_{i \in I} d_{i}, \ && \forall i \in I \\
    &&& \eta \geq 0 \\
    &&& y_{j} \in \{0, 1\}, \ && \forall j \in J.
\end{align*}
```

The primary constraint ensures that the sum of the capacities at all opened facilities is at least the total demand from all customers. This is a vital requirement that helps avoid any infeasibilities in the subproblem.

### Subproblem

For a given set of open facilities specified by the solution $(\bar{y}, \bar{\eta})$ from the master problem, the subproblem focuses on the optimal distribution of customer demand. The subproblem is a Linear Program (LP) that minimizes the transportation costs associated with satisfying customer demands, under the facility openings proposed by $\bar{y}$. The subproblem is described as follows:

```math
\begin{align*}
    \psi(\bar{y}) ={} & \min\ && \sum_{i \in I} \sum_{j \in J} c_{ij}\, x_{ij}\\
    & \text{s.t.} && \sum_{j \in J} x_{ij} \geq d_{i},\ && \forall i \in I \\
    &&& -\sum_{i \in I} x_{ij} \geq -u_{j}\, \bar{y}_{j},\ && \forall j \in J \\
    &&& x_{ij} \geq 0, \ && \forall i \in I, \ j \in J.
\end{align*}
```

The subproblem's role is to evaluate the cost-effectiveness of the candidate solution $\bar{y}$ from the master problem. If the subproblem's optimal cost, $\psi^\ast(\bar{y})$, exceeds $\bar{\eta}$, we must adjust our master problem through an optimality cut derived from the dual solution of the subproblem. Otherwise, if $\psi^\ast(\bar{y}) \leq \bar{\eta}$ for an incumbent $\bar{y}$, then $\bar{y}$ is an optimal decision for the original problem.

### Generating Optimality Cuts

After solving the master problem, an optimality cut is generated based on the optimal dual variable values from the subproblem. Let dual variables $\mu$ and $\nu$ correspond to the constraints related to demand satisfaction and facility capacity, respectively. The dual formulation is:

```math
\begin{align*}
    & \max \ && \sum_{i \in I} d_{i}\, \mu_{i} - \sum_{j \in J} u_{j} \bar{y}_{j}\, \nu_{j} \\
    & \text{s.t.} && \mu_{i} - \nu_{j} \leq c_{ij}, \ && \forall i \in I, \ j \in J \\
    &&& \mu_{i} \geq 0, \ && \forall i \in I \\
    &&& \nu_{j} \geq 0, \ && \forall j \in J.
\end{align*}
```

With $\mu^\ast$ and $\nu^\ast$ being the optimal values of the dual variables, the optimality cut is formulated as:

```math
\eta \geq \sum_{i \in I} \mu^\ast_{i} d_{i} - \sum_{j \in J} \nu^\ast_{j} u_{j}\, y_{j}
```

This optimality cut is then added to the master problem to refine the search space of the solution and guide the optimization process closer to the optimum of the original problem.

## Implementation

We use the Gurobi Optimizer and its Python interface `gurobipy` to solve CFLP instances.
An MIP model is created to represent the master problem of the CFLP. To incorporate Benders decomposition, we use the `callback` feature of Gurobi. This allows dynamic addition of Benders cuts to the model during the internal branch-and-bound optimization process.

The callback function is defined and invoked by Gurobi at predefined points as the MIP is solved. Within this callback:

1. The current solution of the master problem is obtained
1. The subproblem is solved to obtain dual variables
1. Optimality cuts are generated using the dual variables
1. Cuts are added to the master problem as lazy constraints

Callbacks allow us to dynamically add constraints during runtime. Optimality cuts can be added as *lazy* constraints, meaning that they are incorporated into the master problem only when they become violated during the branching process. This approach significantly reduces the branching tree size, resulting in faster convergence


### Project Files Overview

This project consists of the following key Python files:

- `data.py`: Responsible for generating synthetic datasets for the CFLP. It allows the customization of problem instances according to user-defined parameters.
- `main.py`: The entry point of the application. It demonstrates the process of setting up the problem, solving it using Benders decomposition, and displaying the results.
- `master_problem.py`: Defines the master MIP using the `gurobipy` interface. It ncludes model setup, variables, objectives, constraints and `callback` function reference.
- `callbacks.py`: Contains Callback class with a `call` method which is invoked during the MIP solving process. It is where the Benders decomposition is implemented by adding optimality cuts.
- `subproblem.py`: Contains the logic for solving the primal LP subproblem. It uses the current solution of the master problem and generates the solution tho the dual variables.
- `requirements.txt`: Python dependencies list for easy reproducibility and environment setup.

## References

- Wentges, P. Accelerating Benders' decomposition for the capacitated facility location problem. *Mathematical Methods of Operations Research* **44**, 267–290 (1996). [DOI:10.1007/BF01194335](https://doi.org/10.1007/BF01194335)
- [Benders Decomposition](https://en.wikipedia.org/wiki/Benders_decomposition)
- [Gurobi Optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/)
- [GurobiPy](https://pypi.org/project/gurobipy/)
