from typing import Optional

import numpy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import expm
from pytransform3d.batch_rotations import quaternion_slerp_batch
from pytransform3d.rotations import q_id
from pytransform3d.trajectories import plot_trajectory
from scipy.spatial.transform import Rotation
from functools import lru_cache
from scipy.linalg import block_diag, cho_solve, cho_factor, norm

data = loadmat('dataset3.mat')

# Question 1 - Q and R matrices
Q0 = np.diag(np.concatenate((data['v_var'], data['w_var'])).flatten())
invQ0 = np.diag(np.concatenate((1/data['v_var'], 1/data['w_var'])).flatten())
R0 = np.diag(data['y_var'].flatten())
invR0 = np.diag((1/data['y_var']).flatten())

theta_vk_i = data['theta_vk_i']
r_i_vk_i = data['r_i_vk_i']
t = data["t"].transpose()

Tgt = np.zeros((theta_vk_i.shape[1], 4, 4))
for i in range(0, theta_vk_i.shape[1]):
    Tgt[i, 0:3, 0:3] = Rotation.from_euler('zyx', theta_vk_i[:,i]).as_matrix()
    Tgt[i, 0:3, 3] = r_i_vk_i[:,i]
    Tgt[i, 3, 3] = 1

@lru_cache
def del_t(k: int) -> int:
    return (t[k] - t[k - 1])[0]

@lru_cache
def Q(k) -> np.ndarray:
    return Q0 * (del_t(k) ** 2)

@lru_cache
def invQ(k) -> np.ndarray:
    return invQ0 / (del_t(k) ** 2)


def numberLandmarks(k) -> int:
    num = 0
    for j in range(0, 20):
        if ykj(k, j)[0] != -1:
            num = num + 1
    return num

@lru_cache
def R(j: int = 0, k: int = 0) -> np.ndarray:
    return R0

@lru_cache
def invR(j: int = 0, k: int = 0) -> np.ndarray:
    return invR0


# Question 4 - Plot visible landmarks
y_k_j = data["y_k_j"].transpose((1, 2, 0))

def plotVisibleLandmarks():
    num_meas = np.sum(1 * (np.sum(y_k_j, axis=2) > 0), 1)
    meas_color = ['green' if n >= 3 else 'red' for n in num_meas]

    plt.scatter(t, num_meas, s=[0.2] * len(num_meas), c=meas_color)
    plt.xlabel(r't [$s$]')
    plt.ylabel(r'Number of Visible Landmarks')
    plt.savefig('number_visible.png', dpi=200)

# Question 5 - Plot visible landmarks
k1 = 1215
k2 = 1714


# Mathematical operations
def cross(r: np.ndarray) -> np.ndarray:
    return np.array([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])


def sunglass(k: int):
    return np.concatenate((data["v_vk_vk_i"][:, k], data["w_vk_vk_i"][:, k]))


def wedge(r: np.ndarray) -> np.ndarray:
    if r.shape[0] == 3:
        return np.array([[0, -r[2], r[1]],
                          [r[2], 0, -r[0]],
                          [-r[1], r[0], 0]])
    else:
        return np.array([[0, -r[5], r[4], r[0]],
                          [r[5], 0, -r[3], r[1]],
                          [-r[4], r[3], 0, r[2]],
                          [0, 0, 0, 0]])


def curly_wedge(r: np.ndarray) -> np.ndarray:
    u = r[0:3]
    v = r[3:6]
    temp1 = np.hstack((wedge(v), wedge(u)))
    temp2 = np.hstack((np.zeros((3, 3)), wedge(v)))

    return np.vstack((temp1, temp2))


def shield(r: np.ndarray) -> np.ndarray:
    if r.shape[0] == 6:
        return np.ndarray([[r[4], 0, r[2], -r[1]],
                          [r[5], -r[2], 0, r[0]],
                          [r[6], r[3], -r[0], 0],
                          [0, 0, 0, 0]])
    else:
        tmp1 = np.hstack((np.eye(3) * r[3], -wedge(r[0:3])))

        return np.vstack((tmp1, np.zeros((1,6))))


# The Hamburger symbol
@lru_cache
def ham(k: int):
    return expm(del_t(k) * wedge(sunglass(k)))


D = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]])

def Ad(M: np.ndarray) -> np.ndarray:
    C = M[0:3, 0:3]
    Jp = M[0:3, 3]

    temp1 = np.hstack((C, wedge(Jp) @ C))
    temp2 = np.hstack((np.zeros((3, 3)), C))
    return np.vstack((temp1, temp2))

@lru_cache
def F(k: int) -> np.ndarray:
    # assert expm(del_t(k) * curly_wedge(sunglass(k))) == Ad(ham(k))

    return Ad(ham(k))


# Initialization
T_prior = np.zeros((k2, 4, 4))
P_prior = np.zeros((k2, 6, 6))
K = np.zeros((k2, 6, 6))
P_post = np.zeros((k2, 6, 6))
T_post = np.zeros((k2, 4, 4))
for k in range(k1, k2):
    T_prior[k] = Tgt[k]
    T_post[k] = Tgt[k]
    P_prior[k] = np.eye(6) * 1e-4
    P_post[k] = np.eye(6) * 1e-4

@lru_cache
def Cvki(k):
    return T_prior[k][0:3,0:3]

@lru_cache
def rvki(k):
    return T_prior[k][3,0:3]

@lru_cache
def p(j, k):
    Ccv = data['C_c_v']
    rho_v_c_v = data['rho_v_c_v'].squeeze()
    rho_i_pj_i = data['rho_i_pj_i'].transpose()

    rpji = Ccv @ (Cvki(k) @ (rho_i_pj_i[j] - rvki(k)) - rho_v_c_v)

    return np.hstack((rpji, 1))


@lru_cache
def Gjk(j: int, k: int) -> np.ndarray:
    return D.transpose() @ shield(T_prior[k] @ p(j, k))


@lru_cache
def G(k: int) -> np.ndarray:
    res = np.empty((0, 6))
    for j in range(0, 20):
        if ykj(k, j)[0] != -1:
            res = np.vstack((res, Gjk(j, k)))
    return res

@lru_cache
def ykj(k: int, j: int) -> np.ndarray:
    return data['y_k_j'][:,k,j]

@lru_cache
def y(k: int) -> np.ndarray:
    v = np.empty
    for j in range(0, 20):
        if ykj(k, j)[0] != -1:
            v = np.concatenate((v, ykj(k, j)))
    return v

@lru_cache
def y_prior_jk(j: int, k: int) -> np.ndarray:
    return D.transpose() @ T_prior[k] @ p(j, k)

@lru_cache
def y_prior(k: int) -> np.ndarray:
    v = np.empty
    for j in range(0, 20):
        v = np.concatenate((v, y_prior_jk(j, k)))
    return v

# Using EKF
def ekf():
    for k in range(k1, k2):
        P_prior[k] = F(k - 1) @ P_post[k - 1] @ F(k - 1).T + Q(k)
        T_prior[k] = ham(k) @ T_post[k - 1]

        P_post[k] = P_prior[k]
        T_post[k] = T_prior[k]
        K[k] = P_prior[k] @ G(k).T @ (G(k) @ P_prior[k] @ G(k).T + R(k)).I
        P_post = (1 - K[k] @ G(k)) @ P_prior[k]
        T_post[k] = numpy.exp((K[k] @ (y(k) - y_prior(k)))) * T_prior[k]

def plotTrajectory(T, k1, k2):
    P = np.zeros((k2 - k1, 7))
    P[:, 0:3] = T[k1:k2,0:3,3]
    for k in range(k1, k2):
        P[k - k1,3:7] = Rotation.from_matrix(T[k,0:3,0:3]).as_quat()

    ax = plot_trajectory(
        P=P, s=0.05, n_frames=100, normalize_quaternions=False, lw=1, c="k")
    ax.set(xlim=(1.5, 3.5), ylim=(1.5, 3.5), zlim=(0, 2))

    plt.show()

# plotTrajectory(Tgt, k1, k2)
# plotTrajectory(T_post, k1, k2)


@lru_cache
def H():
    K = (k2 - k1 + 1)
    H = np.zeros((6 * K, 6 * K))
    for k in range(0, K):
        H[k * 6:(k + 1) * 6, k * 6:(k + 1) * 6] = np.eye(6)
    for k in range(0, K - 1):
        H[(k + 1) * 6:(k + 2) * 6, k * 6:(k + 1) * 6] = -F(k + k1)
    for k in range(0, K - 1):
        g = G(k + k1)
        tmp = np.zeros((g.shape[0], 6 * K))
        tmp[0:g.shape[0], k * 6:(k + 1) * 6] = g
        H = np.vstack((H, tmp))

    return H

def invW():
    tempQ = block_diag(*[invQ(k) for k in range(k1, k2)])
    tempR = []
    for k in range(k1, k2):
        for i in range(0, numberLandmarks(k)):
            tempR.append(invR())
    tempR = block_diag(*tempR)

    return block_diag(np.linalg.inv(P_prior[k1]), tempQ, tempR)

def evk(k: int, x):

    pass

def eyk(k: int, x):
    pass

def e(x):
    tmp = []
    for k in range(k1, k2):
        tmp.append(evk(k, x))
    for k in range(k1, k2):
        tmp.append(eyk(k, x))
    return np.vstack(*tmp)

def GaussNewtonUpdate():
    A = H().transpose @ invW() @ H()
    b = H().transpose @ invW() @ e(xop)

    update = cho_solve(cho_factor(A), b)
    eps = norm(update)
    # delta_x_star = A / b
    pass

invW()