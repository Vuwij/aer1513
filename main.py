import numpy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import expm
from pytransform3d.batch_rotations import quaternion_slerp_batch
from pytransform3d.rotations import q_id
from pytransform3d.trajectories import plot_trajectory
from scipy.spatial.transform import Rotation

data = loadmat('dataset3.mat')

# Question 1 - Q and R matrice
Q0 = np.diag(np.concatenate((data['v_var'], data['w_var'])).flatten())
R0 = np.diag(data['y_var'].flatten())

theta_vk_i = data['theta_vk_i']
r_i_vk_i = data['r_i_vk_i']

Tgt = np.zeros((theta_vk_i.shape[1], 4, 4))
for i in range(0, theta_vk_i.shape[1]):
    Tgt[i, 0:3, 0:3] = Rotation.from_euler('zyx', theta_vk_i[:,i]).as_matrix()
    Tgt[i, 0:3, 3] = r_i_vk_i[:,i]
    Tgt[i, 3, 3] = 1


def Q(k: int) -> np.ndarray:
    return Q0


def R(k: int) -> np.ndarray:
    return R0


# Question 4 - Plot visible landmarks
y_k_j = data["y_k_j"].transpose((1, 2, 0))
t = data["t"].transpose()

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
def cross(r: np.ndarray) -> np.matrix:
    return np.matrix([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])


def sunglass(k: int):
    return (np.concatenate((data["v_vk_vk_i"][:, k], data["w_vk_vk_i"][:, k])).transpose()[np.newaxis]).transpose()


def v():
    pass


def wedge(r: np.ndarray) -> np.matrix:
    if r.shape[0] == 3:
        return np.matrix([[0, -r[2], r[1]],
                          [r[2], 0, -r[0]],
                          [-r[1], r[0], 0]])
    else:
        return np.matrix([[0, -r[5], r[4], r[0]],
                          [r[5], 0, -r[3], r[1]],
                          [-r[4], r[3], 0, r[2]],
                          [0, 0, 0, 0]])


def curly_wedge(r: np.ndarray) -> np.ndarray:
    u = r[0:3]
    v = r[3:6]
    temp1 = np.hstack((wedge(v), wedge(u)))
    temp2 = np.hstack((np.zeros((3, 3)), wedge(v)))

    return np.vstack((temp1, temp2))


def shield(r: np.ndarray) -> np.matrix:
    return np.matrix([[r[4], 0, r[2], -r[1]],
                      [r[5], -r[2], 0, r[0]],
                      [r[6], r[3], -r[0], 0],
                      [0, 0, 0, 0]])


def del_t(k: int) -> int:
    return (t[k] - t[k - 1])[0]


# The Hamburger symbol
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

def F(k: int) -> np.matrix:
    # assert expm(del_t(k) * curly_wedge(sunglass(k))) == Ad(ham(k))

    return Ad(ham(k))


# Initialization
T_prior = np.zeros((k2, 4, 4))
P_prior = np.zeros((k2, 6, 6))
K = np.zeros((k2, 6, 6))
P_post = np.zeros((k2, 6, 6))
T_post = np.zeros((k2, 4, 4))

def Cvki(k):
    return T_prior[k][0:2,0:2]

def rvki(k):
    return T_prior[k][3,0:2]

def p(j, k):
    Ccv = data['C_c_v']
    rho_v_c_v = data['rho_v_c_v']
    rho_i_pj_i = data['rho_i_pj_i']

    return Ccv @ (Cvki(k) @ (rho_i_pj_i - rvki(k)) - rho_v_c_v)


def Gjk(j: int, k: int):
    return D.transpose() @ shield(T_prior[k] @ p(j, k))


def G(k: int):
    res = np.empty
    for j in range(1, 20):
        res = np.concatenate((res, Gjk(j, k)))
    return res


def ykj(k: int, j: int) -> np.ndarray:
    return data['y_k_j'][k][j]


def y(k: int) -> np.ndarray:
    v = np.empty
    for j in range(1, 20):
        v = np.concatenate((v, ykj(k, j)))
    return v


def y_prior_jk(j: int, k: int) -> np.ndarray:
    return D.transpose() @ T_prior[k] @ p(j, k)


def y_prior(k: int) -> np.ndarray:
    v = np.empty
    for j in range(1, 20):
        v = np.concatenate((v, y_prior_jk(j, k)))
    return v

# Using EKF
for k in range(k1, k2):
    P_prior[k] = F(k - 1) @ P_post[k - 1] @ F(k - 1).T + Q(k)
    T_prior[k] = ham(k) @ T_post[k - 1]

    P_post[k] = P_prior[k]
    T_post[k] = T_prior[k]
    # K[k] = P_prior[k] @ G(k).T @ (G(k) @ P_prior[k] @ G(k).T + R(k)).I
    # P_post = (1 - K[k] @ G(k)) @ P_prior[k]
    # T_post[k] = numpy.exp((K[k] @ (y(k) - y_prior(k)))) * T_prior[k]

def plotTrajectory(T, k1, k2):
    P = np.zeros((k2 - k1, 7))
    P[:, 0:3] = T[k1:k2,0:3,3]
    for k in range(k1, k2):
        P[k - k1,3:7] = Rotation.from_matrix(T[k,0:3,0:3]).as_quat()

    ax = plot_trajectory(
        P=P, s=0.05, n_frames=100, normalize_quaternions=False, lw=1, c="k")
    ax.set(xlim=(1.5, 3.5), ylim=(1.5, 3.5), zlim=(0, 2))

    plt.show()

plotTrajectory(Tgt, k1, k2)
plotTrajectory(T_post, k1, k2)

def e(x_op: np.ndarray):

    pass
xop

A = H.transpose @ np.linalg.inv(W) @ H
b = H.transpose @ np.linalg.inv(W) @ e(xop)

delta_x_star = A / b