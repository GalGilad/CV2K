import numpy as np
from numpy.random import randn, rand, randint
from sklearn.preprocessing import normalize
from datetime import timedelta
import time

EPSILON = np.finfo(np.float32).eps
    

# calculate KL-divergence
def calc_kld(V, W, H, O):
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH).ravel()
    V_O = np.multiply(O, V).ravel()
    indices = V_O > EPSILON
    WH_relev = WH_O[indices]
    V_relev = V_O[indices]
    WH_relev[WH_relev == 0] = EPSILON
    div = np.divide(V_relev, WH_relev)
    kld = np.dot(V_relev, np.log(div))
    kld += np.sum(WH_relev) - np.sum(V_relev)

    return kld


# calculate frobenius
def calc_euc(V, W, H, O):
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH).ravel()
    V_O = np.multiply(O, V).ravel()
    euc = .5 * np.sum((V_O - WH_O) ** 2)

    return euc
    
    
# calculate Itakura-Saito
def calc_is(V, W, H, O):
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH).ravel()
    V_O = np.multiply(O, V).ravel()
    indices = V_O > EPSILON
    WH_relev = WH_O[indices]
    V_relev = V_O[indices]
    WH_relev[WH_relev == 0] = EPSILON
    div = np.divide(V_relev, WH_relev)
    IS = np.sum(div - np.log(div))

    return IS


def calc_train_error(V, W, H, O, obj='kl'):
    if obj == 'kl':
        return calc_kld(V, W, H, O)
    elif obj == 'euc':
        return calc_euc(V, W, H, O)
    else:
        return calc_is(V, W, H, O)


def calc_val_error(V, W, H, O):
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH)
    V_O = np.multiply(O, V)
    error = np.sum(np.abs(V_O - WH_O))

    return error
    

# Fast KL-divergence update, using: Fast Bregman Divergence NMF using Taylor Expansion and Coordinate Descent, Li 2012
def sBCD_update(V, W, H, O, reg=0, obj='kl'):
    n, m = V.shape
    K = W.shape[1]
    V_tag = np.dot(W, H)
    E = np.subtract(V, V_tag)

    if obj == 'kl':
        B = np.divide(1, V_tag) * O
    elif obj == 'euc':
        B = np.ones((V.shape)) * O
    else:  # obj == 'is'
        B = np.divide(1, V_tag ** 2) * O

    for k in range(K):
        V_k = np.add(E, np.dot(W[:, k].reshape((n, 1)), H[k, :].reshape((1, m))))
        B_V_k = B * V_k
        # update H
        H[k] = np.maximum(1e-16, (np.dot(B_V_k.T, W[:, k])) / (np.dot(B.T, W[:, k] ** 2)))
        # update W
        W[:, k] = np.maximum(1e-16, (reg + np.dot(B_V_k, H[k])) / (reg * W[:, k] + np.dot(B, H[k] ** 2)))
        E = np.subtract(V_k, np.dot(W[:, k].reshape((n, 1)), H[k, :].reshape((1, m))))

    return W, H


def init_factor_matrices(V, rank, O, eps, obj, reg, best_of=100):
    n, m = V.shape
    d = np.sqrt(np.mean(V[O==1]) / rank)
    n_iters = 50
    W, H, errors = [], [], []

    for i in range(best_of):
        W.append(np.absolute(d * randn(n, rank)))
        H.append(np.absolute(d * randn(rank, m)))
        W[i], H[i], error = optimize(V, W[i], H[i], O, maxiter=n_iters, eps=eps, obj=obj, init=True, reg=reg)
        errors.append(error)

    err_argsort = np.argsort(np.array(errors))
    best_of = int(best_of / 2)
    err_argsort = err_argsort[:best_of]
    while best_of > 2:
        W = np.array(W)[err_argsort]
        H = np.array(H)[err_argsort]
        errors = []
        for i in range(best_of):
            W[i], H[i], error = optimize(V, W[i], H[i], O, maxiter=n_iters, eps=eps, obj=obj, init=True, reg=reg)
            errors.append(error)
        err_argsort = np.argsort(np.array(errors))
        best_of = int(best_of / 2)
        err_argsort = err_argsort[:best_of]

    best = np.argmin(errors)
    return W[best], H[best]


def optimize(V, W, H, O, maxiter=5000, eps=1e-8, obj='kl', init=False, reg=0):
    start = time.time()
    converged = False
    previous_error = calc_train_error(V, W, H, O, obj=obj)
    n_iters = 0
    current_error = 0
    check_conv_intervals = 10
    while not converged and n_iters < maxiter:
        n_iters += 1
        # one step
        W, H = sBCD_update(V, W, H, O, reg=reg, obj=obj)
        if n_iters % check_conv_intervals == 0:
            current_error = calc_train_error(V, W, H, O, obj=obj)
            converged = ((previous_error - current_error) / previous_error) < eps
            previous_error = current_error
    end = time.time()
    if not init: print("converged. took %d iterations, runtime was " % (n_iters) + str(timedelta(seconds=end - start)))

    return W, H, current_error

# run nmf: V ~ WH
def crossval_nmf(V, rank, maxiter=2000, eps=1e-8, obj='kl', reg=0, fraction=.01, best_of=100):
    np.random.seed()

    n, m = V.shape
    relative_fraction = fraction
    O = rand(n, m) > relative_fraction

    W, H = init_factor_matrices(V, rank, O, eps, obj, reg=reg, best_of=best_of)
    W, H, _ = optimize(V, W, H, O, maxiter=maxiter, eps=eps, obj=obj, reg=reg)
    print("current rank: {}, fraction: {}".format(rank, relative_fraction))

    H, norm_H = normalize(H, norm='l1', axis=1, return_norm=True)
    W = np.dot(W, np.diag(norm_H))
    V_tag = normalize(V.astype(float), norm='l1', axis=1)
    W_tag = normalize(W, norm='l1', axis=1)
    val_error = calc_val_error(V_tag, W_tag, H, ~O) / np.sum(~O)

    return W, H, val_error