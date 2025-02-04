from data import read_dataset
from master_problem import solve_CFLP

DATA_DIR = "facility-location/data/"

if __name__ == "__main__":
    # file path to the dataset
    datafile = DATA_DIR + "p1"

    # Read the dataset from the file
    data = read_dataset(datafile)

    # Solve the Capacitated Facility Location Problem (CFLP) model and obtain the optimal solution
    solution = solve_CFLP(data)

    # Print solution information
    print("Objective value:    ", solution.objective_value)
    print("Open facilities:    ", [j for j in data.J if solution.locations[j] > 0.5])
    print("Solution time (sec):", solution.solution_time)
    print("No. of optimality cuts generated:", solution.num_cuts_mip)
    print("No. of optimality cuts generated (at node relaxation):", solution.num_cuts_rel)
    print("No. of explored Branch-and-Bound nodes:", solution.num_bnb_nodes)
