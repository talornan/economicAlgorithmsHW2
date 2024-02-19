import cvxpy as cp


def maximize_product_allocation(t):
    """
    Finds the allocation that maximizes the product of values, as a function of t.

    Args:
        t (float): Parameter between 0 and 1.

    Returns:
        float: Optimal variable value.

    Examples:
        >>> maximize_product_allocation(2/3)
        0.75

        >>> maximize_product_allocation(1/2)
        1.0
        >>> maximize_product_allocation(4/5)
        0.62
        >>> maximize_product_allocation(1)
        0.5
    """
    # Define the variable x
    x = cp.Variable(1)

    # Define the objective function to maximize the product of values
    objective = cp.Maximize(cp.log(x) + cp.log(1 - t * x))

    # Define the constraints (0 <= x <= 1)
    constraints = [x >= 0, x <= 1]

    # Create the optimization problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem using a suitable solver (e.g., SCS)
    prob.solve(solver=cp.SCS)

    # Round the value to 2 decimal places
    optimal_variable = round(float(x.value[0]), 2)

    return optimal_variable


if __name__ == "__main__":
    import doctest

    doctest.testmod()
