from data import Data
from gurobipy import GRB, quicksum
from sub_problem import solve_subproblem


class Callback:
    """
    Callback class for implementing Benders' decomposition optimality cuts in the
    Capacitated Facility Location Problem (CFLP). This class is designed to interface
    with the Gurobi optimizer and is invoked during the Mixed-Integer Programming
    (MIP) solution process. The class method is called at MIPSOL (MIP solution) callback
    events to assess newly found integer feasible solutions and to inject lazy optimality
    cuts as needed. These cuts are used to refine the solution space and guide the solver
    towards optimality by eliminating sub-optimal solutions.

    For more information on callback codes, please refer to the Gurobi documentation:
    https://www.gurobi.com/documentation/current/refman/cb_codes.html
    """

    def __init__(self, dat: Data, y, eta):
        """
        Initialize the Callback object.

        Args:
            dat (Data): The data object containing problem data.
            y (Var): The location variables.
            eta (Var): The cost variable.
        """

        self.dat = dat
        self.y = y
        self.eta = eta

        self.num_cuts_mip = 0  # number of optimality cuts added
        self.num_cuts_rel = 0  # number of optimality cuts added (at node relaxation)

    def __call__(self, mod, where):
        """
        Callback entry point: call lazy constraints routine when new
        solutions are found.

        Args:
            mod (Model): The Gurobi model object.
            where (int): The callback event code.

        Returns:
            None
        """

        # Check if an integer feasible solution has been found
        if where == GRB.Callback.MIPSOL:
            # Get the current solution
            y_values = mod.cbGetSolution(self.y)
            eta_value = mod.cbGetSolution(self.eta)

            # Solve the subproblem and obtain its dual information
            obj, mu, nu = solve_subproblem(self.dat, y_values)

            # Add an optimality cut if the subproblem objective value is greater than
            # the estimated cost (eta)
            if obj > eta_value:
                self.add_optimality_cut(mod, mu, nu)
                self.num_cuts_mip += 1

        # Currently exploring a MIP node
        elif where == GRB.Callback.MIPNODE:
            # Get the number of nodes explored
            node_count = mod.cbGet(GRB.Callback.MIPNODE_NODCNT)

            # Message when leaving the root node
            if node_count == 1:
                print("Completed solving the root node. Proceeding with branch-and-bound...")

            status = mod.cbGet(GRB.Callback.MIPNODE_STATUS)
            # Check if the node is optimal
            if status != GRB.OPTIMAL:
                # Relaxation solution for the current node is not valid meaning either infeasible or unbounded
                #then don't use the the values for creating cuts
                #So, it goes to return and exists and does not execute any of the following lines
                return

            # Get the current solution (lp-relaxation)
            y_values = mod.cbGetNodeRel(self.y)
            eta_value = mod.cbGetNodeRel(self.eta)

            # Solve the subproblem and obtain its dual information
            obj, mu, nu = solve_subproblem(self.dat, y_values)

            # Add an optimality cut if the subproblem objective value is greater than
            # the estimated cost (eta)
            if obj > eta_value:
                self.add_optimality_cut(mod, mu, nu)
                self.num_cuts_rel += 1

    def add_optimality_cut(self, mod, mu, nu):
        """
        Add an optimality cut to the model as a lazy constraint.

        Optimality cuts are used to eliminate sub-optimal regions in the solution space by introducing
        additional constraints based on dual variable values obtained from solving the subproblem.

        Args:
            mod (Model): The Gurobi model to which the optimality cut is added.
            mu (list): Dual variable values associated with demand constraints.
            nu (list): Dual variable values associated with capacity constraints.

        Returns:
            None
        """

        # Construct the right-hand side (rhs) of the optimality cut which consists of the product sum
        # of demands and dual variables for demand constraints ('mu'), minus the product sum of
        # capacities, dual variables for capacity constraints ('nu'), and the location variables 'y'.
        rhs = quicksum(self.dat.demands[i] * mu[i] for i in self.dat.I)
        rhs += quicksum(self.dat.capacities[j] * nu[j] * self.y[j] for j in self.dat.J)

        # Add the optimality cut to the model as a lazy constraint. The cut ensures that the cost represented
        # by 'eta' is no less than the cost identified by the dual values.
        # Lazy constraints are added to the model only when they are violated by the current integer solution.
        # This helps in reducing the size of the model and speeds up the solving process.
        mod.cbLazy(self.eta >= rhs)
