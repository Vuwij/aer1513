import numpy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import expm

data = loadmat('dataset3.mat')

# Question 1 - Q and R matrices
Q = np.diag(np.concatenate((data['v_var'], data['w_var'])) .flatten())
R = np.diag(data['y_var'].flatten())

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
k1 = 1215
k2 = 1714

# Mathematical operations
def cross(r: np.ndarray) -> np.matrix:
    return np.matrix([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])

def sunglass(k: int):
    return np.concatenate((data["w_vk_vk_i"][:,k], data["v_vk_vk_i"][:,k]))

def v():

    pass

def wedge(r: np.ndarray) -> np.matrix:
    return np.matrix([[0, -r[5], r[4], r[0]],
                      [r[5], 0, -r[3], r[1]],
                      [-r[4], r[3], 0, r[2]],
                      [0, 0, 0, 0]])

def shield(r: np.ndarray) -> np.matrix:
    return np.matrix([[r[4], 0, r[2], -r[1]],
                      [r[5], -r[2], 0, r[0]],
                      [r[6], r[3], -r[0], 0],
                      [0, 0, 0, 0]])


def del_t(k: int):
    return t[k] - t[k-1]

# The Hamburger symbol
def ham(k: int):
    return expm(del_t(k) @ wedge(sunglass(k)))


D = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]])

def p(j, k):
    Ccv = data['C_c_v']
    rho_v_c_v = data['rho_v_c_v']

    return Ccv @ (Cvi(k) @ (rho - rv1) - rho_v_c_v)


def G(k: int):


    return D.T @ ()


# Initialization
T_prior = np.empty((k2,4,4))
P_prior = np.empty((k2,6,6))
K = np.empty((k2,6,6))
P_post = np.empty((k2,6,6))
T_post = np.empty((k2,6,6))


T_prior[k-1] = ham(k-1)

for k in range(k1, k2):
    P_prior[k] = F[k-1] @ P[k-1] @ F[k-1].T + Qk
    T_prior[k] = ham(k) @ T[k-1]
    K[k] = P_prior[k] @ G[k].T @ (G[k] @ P_prior[k] @ G[k].T + R[k]).I
    P_post = (1 - K[k] @ G[k]) @ P_prior[k]
    T_post[k] = numpy.exp((K[k] @ (y[k] - y[k]))) * T_prior[k]

