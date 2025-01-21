# Import PuLP library
from pulp import LpMaximize, LpProblem, LpVariable

# Create a Linear Program
prob = LpProblem("Maximize_Profit", LpMaximize)

# Define decision variables
x = LpVariable('x', lowBound=0, cat='Continuous')  # Units of product A
y = LpVariable('y', lowBound=0, cat='Continuous')  # Units of product B

# Objective Function: Maximize profit
prob += 5 * x + 4 * y, "Total Profit"

# Constraints
prob += 2 * x + 3 * y <= 100, "Labor Constraint"
prob += 3 * x + 2 * y <= 120, "Material Constraint"

# Solve the problem
prob.solve()

# Display the results
print(f"Status: {prob.status}")
print(f"Optimal number of Product A (x): {x.varValue}")
print(f"Optimal number of Product B (y): {y.varValue}")
print(f"Maximum Profit: ${prob.objective.value()}")

