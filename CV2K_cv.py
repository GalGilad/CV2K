import numpy as np
from numpy.random import randn, rand
from sklearn.preprocessing import normalize
import datetime
import time

EPSILON = np.finfo(np.float32).eps


def calc_kld(V, W, H, O):
    """
    Calculates KL-divergence (beta=1)
    :param V: input matrix (n x m)
    :param W: exposure matrix (n x k)
    :param H: signature matrix (k x m)
    :param O: binary mask (n x m)
    :return: KL-divergence
    """
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH).ravel()
    V_O = np.multiply(O, V).ravel()
    indices = V_O > EPSILON
    WH_relev = WH_O[indices]
    V_relev = V_O[indices]
    WH_relev[WH_relev == 0] = EPSILON
    div = np.divide(V_relev, WH_relev)
    kld = np.dot(V_relev, np.log(div))
    kld += np.sum(WH_O) - np.sum(V_O)

    return kld


def calc_euc(V, W, H, O):
    """
    Calculates euclidean distance (beta=2)
    :param V: input matrix (n x m)
    :param W: exposure matrix (n x k)
    :param H: signature matrix (k x m)
    :param O: binary mask (n x m)
    :return: euclidean distance
    """
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH).ravel()
    V_O = np.multiply(O, V).ravel()
    euc = .5 * np.sum((V_O - WH_O) ** 2)

    return euc


def calc_is(V, W, H, O):
    """
    Calculates Itakura-Saito (beta=0)
    :param V: input matrix (n x m)
    :param W: exposure matrix (n x k)
    :param H: signature matrix (k x m)
    :param O: binary mask (n x m)
    :return: Itakura-Saito divergence
    """
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
    """
    Calculates train error, directs to required beta divergence function
    :param V: input matrix (n x m)
    :param W: exposure matrix (n x k)
    :param H: signature matrix (k x m)
    :param O: binary mask (n x m)
    :param obj: optimization objective (euclidean / KL-divergence / Itakura-Saito)
    :return: train error
    """
    if obj == 'kl':
        return calc_kld(V, W, H, O)
    elif obj == 'euc':
        return calc_euc(V, W, H, O)
    else:
        return calc_is(V, W, H, O)


def calc_validation_error(V, W, H, O):
    """
    Calculates validation error
    :param V: input matrix (n x m)
    :param W: exposure matrix (n x k)
    :param H: signature matrix (k x m)
    :param O: binary mask (n x m), considering global context this will be the ~O matrix - masked cells
    :return: validation error
    """
    WH = np.dot(W, H)
    WH_O = np.multiply(O, WH)
    V_O = np.multiply(O, V)
    error = np.sum(np.abs(V_O - WH_O))

    return error


def sBCD_update(V, W, H, O, reg=0, obj='kl'):
    """
    Updates towards V * O ~ WH
    Fast beta-divergence update, using: Fast Bregman Divergence NMF using Taylor Expansion and Coordinate Descent,
    Li 2012
    :param V: input matrix (n x m)
    :param W: exposure matrix (n x k)
    :param H: signature matrix (k x m)
    :param O: binary mask (n x m)
    :param reg: regularization parameter
    :param obj: optimization objective (euclidean / KL-divergence / Itakura-Saito)
    :return: updated W (n x k) and H (k x m) matrices
    """
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
    """
    Initializes W (n x k) and H (k x m) matrices. This initialization scheme starts with best_of (=100) W and H matrices,
    performs 50 optimization steps on each pair, then proceeds with the best_of / 2 (=50) pairs with the lowest error,
    etc. until we are left with the best pair. This attempts to deal with the issue of convergence to local minima.
    :param V: input matrix (n x m)
    :param rank: rank of factorization
    :param O: binary mask (n x m)
    :param eps: small number, used for convergence criterion
    :param obj: optimization objective (euclidean / KL-divergence / Itakura-Saito)
    :param reg: regularization parameter
    :param best_of: initial number of pairs, default is 100
    :return: initialized W (n x k) and H (k x m) matrices
    """
    n, m = V.shape
    d = np.sqrt(np.mean(V[O == 1]) / rank)
    n_iters = 50
    W, H, errors = [], [], []

    for i in range(best_of):
        W.append(np.absolute(d * randn(n, rank)))
        H.append(np.absolute(d * randn(rank, m)))
        W[i], H[i], error = optimize(V, W[i], H[i], O, maxiter=n_iters, eps=eps, obj=obj, init=True, reg=reg)
        errors.append(error)

    W, H = np.array(W), np.array(H)
    best_of = best_of // 2
    err_argsort = np.argsort(errors)
    err_argsort = err_argsort[:best_of]
    arg_of_best = err_argsort[0]

    while best_of > 2:
        W = W[err_argsort]
        H = H[err_argsort]
        errors = []
        for i in range(best_of):
            W[i], H[i], error = optimize(V, W[i], H[i], O, maxiter=n_iters, eps=eps, obj=obj, init=True, reg=reg)
            errors.append(error)
        best_of = best_of // 2
        err_argsort = np.argsort(errors)
        err_argsort = err_argsort[:best_of]
        arg_of_best = err_argsort[0]

    return W[arg_of_best], H[arg_of_best]


def optimize(V, W, H, O, maxiter=5000, eps=1e-8, obj='kl', init=False, reg=0):
    """
    V * O ~ WH optimization
    :param V: input matrix (n x m)
    :param W: initialized exposure matrix (n x k)
    :param H: initialized signature matrix (k x m)
    :param O: binary mask (n x m)
    :param maxiter: maximum number of iterations - NMF steps - before forced break
    :param eps: small number, used for convergence criterion
    :param obj: optimization objective (euclidean / KL-divergence / Itakura-Saito)
    :param init: is the optimization part of the initialization scheme?
    :param reg: regularization parameter
    :return: final W (n x k) and H (k x m) matrices, train error
    """
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
            converged = (previous_error - current_error) / previous_error < eps
            # converged = np.abs(previous_error - current_error) / previous_error < eps
            previous_error = current_error
    end = time.time()
    if not init: print("converged. took %d iterations, runtime was " % (n_iters) + str(datetime.timedelta(
        seconds=end - start)))

    return W, H, current_error


def crossval_nmf(V, rank, maxiter=2000, eps=1e-8, obj='kl', reg=0, fraction=.01):
    """
    run NMF: V * O ~ WH ; for CV2K standard version
    :param V: input matrix (n x m)
    :param rank: rank of factorization (k)
    :param maxiter: maximum number of iterations - NMF steps - before forced break
    :param eps: small number, used for convergence criterion
    :param obj: optimization objective (euclidean / KL-divergence / Itakura-Saito)
    :param reg: regularization parameter
    :param fraction: size of validation set, imputation fraction
    :return: W (n x k) and H (k x m) matrices, imputation error
    """
    np.random.seed()
    n, m = V.shape
    relative_fraction = fraction
    # initialize binary matrix O with ~V.size*fraction masking cells
    O = rand(n, m) > relative_fraction

    W, H = init_factor_matrices(V, rank, O, eps, obj, reg=reg)
    W, H, _ = optimize(V, W, H, O, maxiter=maxiter, eps=eps, obj=obj, reg=reg)
    print("current rank: {}, fraction: {}".format(rank, relative_fraction))

    H, norm_H = normalize(H, norm='l1', axis=1, return_norm=True)
    W = np.dot(W, np.diag(norm_H))
    V_tag = normalize(V.astype(float), norm='l1', axis=1)
    W_tag = normalize(W, norm='l1', axis=1)
    imputation_error = calc_validation_error(V_tag, W_tag, H, ~O) / np.sum(~O)

    return W, H, imputation_error