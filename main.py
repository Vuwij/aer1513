from typing import Optional

import numpy
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
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
y_k_j = data["y_k_j"].transpose((1, 2, 0))
P_prior_0 = np.eye(6) * 1e-4
T_prior_0 = np.zeros((4,4))

def cross(r: np.ndarray) -> np.ndarray:
    return np.array([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])

def angaxis_to_C(p: np.ndarray):
    phi = np.linalg.norm(p)
    a = p / phi
    C = np.cos(phi) * np.eye(3) + (1 - np.cos(phi)) * (np.expand_dims(a,1) @ np.expand_dims(a,0)) - np.sin(phi) * cross(a)
    return C

def rotation_translation_to_T(C: np.ndarray, t: np.ndarray):
    T = np.zeros((4, 4))
    T[0:3, 0:3] = C
    T[0:3, 3] = -C @ t
    T[3, 3] = 1
    return T

def T_to_rotation_translation(T: np.ndarray):
    C = T[0:3,0:3]
    r = - C.T @ T[0:3, 3]
    return C, r

test = np.array([[1,2,3,4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4]
                 ])

Ccv = data['C_c_v']
rho_v_c_v = data['rho_v_c_v']
Tcv = np.vstack((np.hstack((Ccv, data['rho_v_c_v'])), np.array([0, 0, 0, 1])))
Tgt = np.zeros((theta_vk_i.shape[1], 4, 4))
for i in range(0, theta_vk_i.shape[1]):
    C = angaxis_to_C(theta_vk_i[:,i])
    Tgt[i] = rotation_translation_to_T(C, r_i_vk_i[:, i])

D = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]])
fu = float(data['fu'])
fv = float(data['fv'])
b = float(data['b'])
cu = float(data['cu'])
cv = float(data['cv'])

# Initialization
k1 = k1_global = 1215
k2 = k2_global = 1714

T_prior = np.zeros((k2 + 1, 4, 4))
P_prior = np.zeros((k2 + 1, 6, 6))
K = np.zeros((k2 + 1, 6, 6))
P_post = np.zeros((k2 + 1, 6, 6))
T_post = np.zeros((k2 + 1, 4, 4))


@lru_cache
def del_t(k: int) -> int:
    return (t[k + 1] - t[k])[0]

@lru_cache
def Q(k) -> np.ndarray:
    return Q0 * (del_t(k) ** 2)

@lru_cache
def invQ(k) -> np.ndarray:
    return invQ0 / (del_t(k) ** 2)

@lru_cache
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

def plotVisibleLandmarks():
    num_meas = np.sum(1 * (np.sum(y_k_j, axis=2) > 0), 1)
    meas_color = ['green' if n >= 3 else 'red' for n in num_meas]

    plt.scatter(t, num_meas, s=[0.2] * len(num_meas), c=meas_color)
    plt.xlabel(r't [$s$]')
    plt.ylabel(r'Number of Visible Landmarks')
    plt.savefig('number_visible.png', dpi=200)

@lru_cache
def sunglass(k: int):
    return np.concatenate((-data["v_vk_vk_i"][:, k], -data["w_vk_vk_i"][:, k])) # TODO check negative sign


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

def v(r: np.ndarray) -> np.ndarray:
    if r.shape[0] == 3:
        return np.array([r[2, 1], r[0, 2], r[1, 0]])
    else:
        return np.array([r[0, 3], r[1, 3], r[2, 3], r[2, 1], r[0, 2], r[1, 0]])
    pass

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
def ham_0(k: int):
    return expm(del_t(k) * wedge(sunglass(k)))

@lru_cache
def ham(k: int):
    return T_prior[k + 1] @ np.linalg.inv(T_prior[k])


def g(p: np.ndarray) -> np.ndarray:
    x = p[0]
    y = p[1]
    z = p[2]
    return 1/z * np.array([fu * x, fv * y, fu * (x - b), fv * y]) + np.array([cu, cv, cu, cv])

def dgdz(p: np.ndarray) -> np.ndarray:
    x = p[0]
    y = p[1]
    z = p[2]

    return np.array([[fu / z, 0, -fu * x / z ** 2],
                     [0, fv / z, -fv * y / z ** 2],
                     [fu / z, 0, -fu * (x - b) / z ** 2],
                     [0, fv / z, -fv * y / z ** 2]
                     ])

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


Ccv = data['C_c_v']
rho_v_c_v = data['rho_v_c_v'].squeeze()
Tcv = rotation_translation_to_T(Ccv, rho_v_c_v)
rho_i_pj_i = data['rho_i_pj_i'].transpose()

@lru_cache
def p(j, k):
    return Tcv @ T_prior[k] @ np.hstack((rho_i_pj_i[j], 1))

    # rpji = Ccv @ (Cvki(k) @ (rho_i_pj_i[j] - rvki(k)) - rho_v_c_v)
    # return np.hstack((rpji, 1))


@lru_cache
def Gjk(j: int, k: int) -> np.ndarray:
    z = p(j, k)
    dzdx = D.transpose() @ Tcv @ shield(T_prior[k] @ np.hstack((rho_i_pj_i[j], 1)))
    return dgdz(D.transpose() @ z) @ dzdx


@lru_cache
def G(k: int) -> np.ndarray:
    res = []
    for j in range(0, 20):
        if ykj(k, j)[0] != -1:
            res.append(Gjk(j, k))
    return np.vstack(res) if res != [] else np.empty((0, 6))

@lru_cache
def ykj(k: int, j: int) -> np.ndarray:
    return data['y_k_j'][:,k,j]

@lru_cache
def y(k: int) -> np.ndarray:
    v = []
    for j in range(0, 20):
        if ykj(k, j)[0] != -1:
            v.append(ykj(k, j))

    return np.vstack(v)

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
    for k in range(k1, k2 + 1):
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
    for k in range(k1, k2 + 1):
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

    tmp = []
    for k in range(0, K):
        g = G(k + k1)
        tmp2 = np.zeros((g.shape[0], 6 * K))
        tmp2[0:g.shape[0], k * 6:(k + 1) * 6] = g
        tmp.append(tmp2)

    H = np.vstack((H, np.vstack(tmp)))

    return H

@lru_cache
def invW():
    tempQ = block_diag(*[invQ(k) for k in range(k1, k2)])
    tempR = []
    for k in range(k1, k2 + 1):
        for i in range(0, numberLandmarks(k)):
            tempR.append(invR())
    tempR = block_diag(*tempR)

    return block_diag(np.linalg.inv(P_prior_0), tempQ, tempR)

@lru_cache
def evk(k: int) -> np.ndarray:
    if k == k1:
        return v(logm(T_prior_0 @ np.linalg.inv(T_prior[k])))

    return v(logm(ham_0(k-1) @ T_prior[k-1] @ np.linalg.inv(T_prior[k])))
    # TODO check ham
@lru_cache
def eyjk(j: int, k: int) -> np.ndarray:
    return ykj(k, j) - g(p(j, k))

@lru_cache
def eyk(k: int):
    tmp = []
    for j in range(0, 20):
        if ykj(k, j)[0] != -1:
            tmp.append(eyjk(j, k))
    return np.hstack(tmp) if tmp != [] else np.empty((0))

@lru_cache
def e():
    tmp = []
    for k in range(k1, k2 + 1):
        tmp.append(evk(k))
    for k in range(k1, k2 + 1):
        tmp.append(eyk(k))
    return np.hstack(tmp)

def GaussNewtonUpdate():
    A = H().T @ invW() @ H()
    b = H().T @ invW() @ e()

    update = cho_solve(cho_factor(A), b)
    eps = update.reshape(int(update.shape[0] / 6), 6)

    tmp = np.linalg.inv(A)
    for k in range(0, k2 - k1 + 1):
        T_prior[k1 + k] = expm(wedge(eps[k])) @ T_prior[k1 + k]
        P_prior[k1 + k] = tmp[k * 6: (k + 1) * 6, k * 6: (k + 1) * 6]

    return norm(eps)

def Initialize(use_previous: bool = False):
    global T_prior_0
    global P_prior_0

    if use_previous and k1 != k1_global:
        T_prior[k1] = T_prior[k1]
        T_prior_0 = T_prior[k1]
        P_prior[k1] = P_prior_0
    else:
        T_prior[k1] = Tgt[k1]
        T_prior_0 = Tgt[k1]
        P_prior[k1] = P_prior_0

    for k in range(k1 + 1, k2 + 1):
        T_prior[k] = ham_0(k - 1) @ T_prior[k - 1]
        P_prior[k] = F(k - 1) @ P_prior[k - 1] @ F(k - 1).T + Q(k - 1)


def BatchOptimize():
    Initialize()

    iter = 0
    eps = np.inf

    while iter < 20 and eps > 1e-5:
        p.cache_clear()
        Gjk.cache_clear()
        G.cache_clear()
        y_prior_jk.cache_clear()
        y_prior.cache_clear()
        H.cache_clear()
        invW.cache_clear()
        evk.cache_clear()
        eyjk.cache_clear()
        eyk.cache_clear()
        e.cache_clear()
        F.cache_clear()
        ham.cache_clear()

        eps = GaussNewtonUpdate()
        iter += 1
        print('Gauss Newton Step: {}  EPS: {}'.format(iter, eps))
    pass

def PlotErrors(plot_name: str):
    error_rx = np.zeros(k2)
    error_ry = np.zeros(k2)
    error_rz = np.zeros(k2)
    var_rx = np.zeros(k2)
    var_ry = np.zeros(k2)
    var_rz = np.zeros(k2)

    error_thetax = np.zeros(k2)
    error_thetay = np.zeros(k2)
    error_thetaz = np.zeros(k2)
    var_thetax = np.zeros(k2)
    var_thetay = np.zeros(k2)
    var_thetaz = np.zeros(k2)

    for k in range(k1, k2):
        C, r = T_to_rotation_translation(T_prior[k])
        Cgt, rgt = T_to_rotation_translation(Tgt[k])

        error_rx[k] = r[0] - rgt[0]
        error_ry[k] = r[1] - rgt[1]
        error_rz[k] = r[2] - rgt[2]

        tmp = v(np.eye(3) - Cgt @ np.linalg.inv(C))
        error_thetax[k] = tmp[0]
        error_thetay[k] = tmp[1]
        error_thetaz[k] = tmp[2]

        var_rx[k] = np.sqrt(P_prior[k][0,0])
        var_ry[k] = np.sqrt(P_prior[k][1,1])
        var_rz[k] = np.sqrt(P_prior[k][2,2])
        var_thetax[k] = np.sqrt(P_prior[k][3,3])
        var_thetay[k] = np.sqrt(P_prior[k][4,4])
        var_thetaz[k] = np.sqrt(P_prior[k][5,5])

    fig = plt.figure()
    fig.set_size_inches(8, 12)

    def subploterror(error_data, variance_data, label, unit):
        plt.plot(t[k1:k2], error_data[k1:k2])
        plt.fill_between(t[k1:k2,0], -3 * variance_data[k1:k2], 3 * variance_data[k1:k2], alpha=0.3)
        plt.xlabel("t [s]")
        plt.ylabel("$\delta$ " + label + " [" + unit + "]")
        pass

    plt.title(plot_name)
    plt.subplot(611)
    subploterror(error_rx, var_rx, 'rx', 'm')
    plt.subplot(612)
    subploterror(error_ry, var_ry, 'ry', 'm')
    plt.subplot(613)
    subploterror(error_rz, var_rz, 'rz', 'm')
    plt.subplot(614)
    subploterror(error_thetax, var_thetax, 'thetax', 'rad')
    plt.subplot(615)
    subploterror(error_thetay, var_thetay, 'thetay', 'rad')
    plt.subplot(616)
    subploterror(error_thetaz, var_thetaz, 'thetaz', 'rad')

    plt.show()
    pass

def SlidingWindow(kappa):
    global k1, k2

    for k in range(k1_global, k2_global - kappa):
        print("k = " + str(k))
        k1 = k
        k2 = k + kappa
        Initialize(True)
        BatchOptimize()

    pass

print("Plotting data points")
plotVisibleLandmarks()
print("Batch Optimization")
# BatchOptimize()
# PlotErrors("Batch Optimization")
print("Sliding Window with Kappa = 50")
SlidingWindow(50)
PlotErrors("Sliding Window 50")
print("Sliding Window with Kappa = 10")
SlidingWindow(10)
PlotErrors("Sliding Window 10")