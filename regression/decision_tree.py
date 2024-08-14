import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

# build a dummy dataset of production cost and profit
dataset = np.array(
    [['Asset Flip', 100, 1000],
     ['Text Based', 500, 3000],
     ['Visual Novel', 1500, 5000],
     ['2D Pixel Art', 3500, 8000],
     ['2D Vector Art', 5000, 6500],
     ['Strategy', 6000, 7000],
     ['First Person Shooter', 8000, 15000],
     ['Simulator', 9500, 20000],
     ['Racing', 12000, 21000],
     ['RPG', 14000, 25000],
     ['Sandbox', 15500, 27000],
     ['Open-World', 16500, 30000],
     ['MMOFPS', 25000, 52000],
     ['MMORPG', 30000, 80000]
     ])

X = dataset[:, 1:2].astype(int)

y = dataset[:, 2].astype(int)

model = DecisionTreeRegressor(random_state=0)

model.fit(X, y)

# test with an invented value
production_cost = 3750
y_predicted = model.predict([[production_cost]])

print(f"Predicted profit: {y_predicted}")

# visualize the data
X_grid = np.arange(min(X), max(X), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array, i.e. to make a column out of the X_grid values
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, model.predict(X_grid), color='blue')
plt.title('Profit to Production Cost (Decision Tree Regression)')
plt.xlabel('Production Cost')
plt.ylabel('Profit')
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

