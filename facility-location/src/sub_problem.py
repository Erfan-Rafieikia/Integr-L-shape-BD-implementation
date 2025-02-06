from data import Data
from gurobipy import GRB, Model, quicksum, tupledict


def _set_params(model: Model) -> None:
    """Set the parameters for the Gurobi solver to suppress console output."""
    model.Params.OutputFlag = 0


def solve_subproblem(dat: Data, facility_open: tupledict, linear_relaxation: bool = True) -> tuple:
    """
    Solve the subproblem for the Facility Location Problem (FLP).

    This function defines and optimizes the subproblem given the facility open decisions.
    It calculates the optimal shipment quantities from facilities to customers minimizing
    the total shipment costs while satisfying demand and capacity constraints.

    Args:
        dat (Data): The input data containing costs, demands, and capacities.
        facility_open (tupledict): A dictionary with facility indices as keys and binary
                                   decisions (1 if open, 0 if closed) as values.
        linear_relaxation (bool): If True, solves the LP relaxation with continuous `x`.
                                  If False, solves the problem with binary `x`.

    Returns:
        tuple: If `linear_relaxation` is True, returns a tuple containing the objective value,
               dual values for demand constraints (mu), and dual values for capacity constraints (nu).
               If `linear_relaxation` is False, returns only the objective value.
    """

    with Model("FLP_Sub") as mod:
        _set_params(mod)

        # Decision variables for shipment quantities
        if linear_relaxation:
            x = mod.addVars(dat.I, dat.J, name="x")
        else:
            x = mod.addVars(dat.I, dat.J, vtype=GRB.BINARY, name="x")

        # Objective: Minimize total shipment costs
        total_cost = quicksum(dat.shipment_costs[i, j] * x[i, j] * dat.demands[i] for i in dat.I for j in dat.J)
        mod.setObjective(total_cost, GRB.MINIMIZE)

        # Constraints: Satisfy demand for each customer
        demand_constraints = mod.addConstrs(
            (quicksum(x[i, j] for j in dat.J) >= 1 for i in dat.I),
            name="Demand",
        )

        # Constraints: Do not exceed capacity for open facilities
        capacity_constraints = mod.addConstrs(
            (
                quicksum(x[i, j] * dat.demands[i] for i in dat.I) <= dat.capacities[j] * facility_open[j]
                for j in dat.J
            ),
            name="Capacity",
        )

        # Optimize the model
        mod.optimize()

        # Retrieve the objective value
        objective_value = mod.ObjVal

        # If solving LP relaxation, return dual values
        if linear_relaxation:
            mu_values = mod.getAttr("pi", demand_constraints)
            nu_values = mod.getAttr("pi", capacity_constraints)
            return objective_value, mu_values, nu_values
        else:
            return objective_value
