from dataclasses import dataclass

import numpy as np


@dataclass
class Data:
    I: np.ndarray  # Customer index list
    J: np.ndarray  # Facility index list
    demands: np.ndarray  # Customer demands
    capacities: np.ndarray  # Facility capacities
    fixed_costs: np.ndarray  # Facility opening costs
    shipment_costs: np.ndarray  # Transportation costs


def word_reader(file_path):
    with open(file_path, "r") as file:
        for line in file:
            for word in line.split():
                yield word


def read_dataset(file_path):
    """
    Reads a dataset for the capacitated facility location problem.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        Data: A Data object containing the instance information.
    """
    word = word_reader(file_path)

    # Read the number of facilities and customers
    num_facilities = int(next(word))
    num_customers = int(next(word))

    # Read facility capacities and fixed costs
    capacities = []
    fixed_costs = []
    for _ in range(num_facilities):
        capacity = int(next(word))
        fixed_cost = int(next(word))
        capacities.append(capacity)
        fixed_costs.append(fixed_cost)

    demands = np.array([float(next(word)) for _ in range(num_customers)])

    # Read transportation costs as an m x n matrix
    shipment_costs = np.array(
        [float(next(word)) for _ in range(num_facilities * num_customers)]
    ).reshape(num_facilities, num_customers)

    shipment_costs = np.transpose(shipment_costs)

    # Convert data to numpy arrays
    I = np.arange(num_customers)  # Customer indices
    J = np.arange(num_facilities)  # Facility indices
    capacities = np.array(capacities)
    fixed_costs = np.array(fixed_costs)

    # Print the loaded data for verification
    print(f"Customer indices (I): {I}")
    print(f"Facility indices (J): {J}")
    print(f"Customer demands: {demands}")
    print(f"Facility capacities: {capacities}")
    print(f"Facility fixed costs: {fixed_costs}")
    print(f"Shipment costs matrix:\n{shipment_costs}")

    return Data(
        I=I,
        J=J,
        demands=demands,
        capacities=capacities,
        fixed_costs=fixed_costs,
        shipment_costs=shipment_costs,
    )
