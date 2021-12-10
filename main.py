import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('dataset3.mat')

# Question 1 - Q and R matrices
Qk = np.diag(np.concatenate((data['v_var'], data['w_var'])) .flatten())
Rk = np.diag(data['y_var'].flatten())

# Question 4 - Plot visible landmarks
y_k_j = data["y_k_j"].transpose((1, 2, 0))
t = data["t"]

num_meas = np.sum(1*(np.sum(y_k_j, axis=2) > 0),1)
meas_color = ['green' if n >= 3 else 'red' for n in num_meas]

plt.scatter(t, num_meas, s=[0.2] * len(num_meas), c=meas_color)
plt.xlabel(r't [$s$]')
plt.ylabel(r'Number of Visible Landmarks')
plt.savefig('number_visible.png', dpi=200)

# Question 5 - Plot visible landmarks
