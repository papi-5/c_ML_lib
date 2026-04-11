import numpy as np
x = np.random.uniform(-3, 2, 1000)
y = x**3 + 2*x + 1 + np.random.normal(0, 0.1, 1000)
np.savetxt('regression_0.csv', np.column_stack([y, x]), delimiter=',')
np.savetxt('regression_1.csv', np.column_stack([x, y]), delimiter=',')