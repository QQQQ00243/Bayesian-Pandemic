import numpy as np


def norm2(x1, x2):
    delta_squared = (x1 - x2)**2
    return np.sqrt(np.sum(delta_squared))


def R_L_deconvolve(obs, H, max_iter=1000, eps=1e-3):
    x_old = np.array(obs)
    H = np.array(H)
    converge = False
    for _ in range(max_iter):
        E = np.dot(H, x_old)
        rel = (obs / E) / np.dot(H.transpose(), np.ones(H.shape[0]))
        x_new = x_old * np.dot(H.transpose(), rel)
        if norm2(x_new, x_old) < eps:
            converge = True
            break
        else:
            x_old = x_new
    if not converge:
        print(f"[Warning] Exit without converging for threshold {eps} with number of iterations \
            {max_iter}. Try increasing the max_iter.")
    return x_new


def R_L_deconvolve_step(obs, H, step_size=0.1, max_iter=1000, eps=1e-3):
    x_old = np.array(obs)
    H = np.array(H)
    converge = False
    for _ in range(max_iter):
        E = np.dot(H, x_old)
        grad = np.dot(H.transpose(), obs/E-np.ones(H.shape[0]))
        x_new = x_old + step_size*grad
        if norm2(x_new, x_old) < eps:
            converge = True
            break
        else:
            x_old = x_new
    if not converge:
        print(f"[Warning] Exit without converging for threshold {eps} with number of iterations \
            {max_iter}. Try increasing the max_iter.")
    return x_new



