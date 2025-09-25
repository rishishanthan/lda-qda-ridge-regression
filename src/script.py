import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pickle
import sys


# ----------------------------
# Helpers
# ----------------------------
def _as_column(z):
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape((-1, 1))
    return z


def _regularize_cov(S, eps=1e-6):
    # Add small ridge to diagonal for numerical stability
    S = np.asarray(S)
    return S + eps * np.eye(S.shape[0])


# ----------------------------
# LDA / QDA
# ----------------------------
def ldaLearn(X, y):
    """
    Inputs:
        X: N x d
        y: N x 1
    Outputs:
        means: d x k (each column is class mean)
        covmat: d x d (shared covariance)
    """
    X = np.asarray(X)
    y = _as_column(y).ravel()
    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    means = np.zeros((d, k))
    for j, c in enumerate(classes):
        Xc = X[y == c]
        means[:, j] = Xc.mean(axis=0)
    # Shared covariance (MLE with 1/N factor)
    covmat = np.cov(X, rowvar=False, bias=True)
    return means, covmat


def qdaLearn(X, y):
    """
    Outputs:
        means: d x k
        covmats: list of k (each d x d)
    """
    X = np.asarray(X)
    y = _as_column(y).ravel()
    classes = np.unique(y)
    d = X.shape[1]
    k = len(classes)
    means = np.zeros((d, k))
    covmats = []
    for j, c in enumerate(classes):
        Xc = X[y == c]
        means[:, j] = Xc.mean(axis=0)
        Sc = np.cov(Xc, rowvar=False, bias=True)
        covmats.append(_regularize_cov(Sc))
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    """
    Returns:
        acc (scalar), ypred (N x 1)
    """
    Xtest = np.asarray(Xtest)
    ytest = _as_column(ytest).ravel()
    classes = np.arange(1, means.shape[1] + 1)  # assume labels are 1..k
    Sigma_inv = inv(_regularize_cov(covmat))
    # Discriminant: g_k(x) = -0.5 (x - mu_k)^T Sigma^{-1} (x - mu_k)
    # (equal priors; terms independent of k drop out)
    N = Xtest.shape[0]
    scores = np.zeros((N, means.shape[1]))
    for j in range(means.shape[1]):
        mu = means[:, j]
        diff = Xtest - mu
        # quadratic form for all rows
        q = np.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)
        scores[:, j] = -0.5 * q
    idx = np.argmax(scores, axis=1)
    ypred = classes[idx].reshape((-1, 1))
    if ytest.size == 0 or (ytest == 0).all():
        acc = 0.0
    else:
        acc = np.mean(ypred.ravel() == ytest.ravel())
    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    """
    Returns:
        acc (scalar), ypred (N x 1)
    """
    Xtest = np.asarray(Xtest)
    ytest = _as_column(ytest).ravel()
    classes = np.arange(1, means.shape[1] + 1)  # assume labels are 1..k
    N = Xtest.shape[0]
    k = means.shape[1]
    scores = np.zeros((N, k))
    for j in range(k):
        mu = means[:, j]
        Sj = _regularize_cov(covmats[j])
        Sj_inv = inv(Sj)
        sign, logdet = np.linalg.slogdet(Sj)
        if sign <= 0:
            # fallback: add more regularization
            Sj = _regularize_cov(Sj, eps=1e-4)
            Sj_inv = inv(Sj)
            sign, logdet = np.linalg.slogdet(Sj)
        diff = Xtest - mu
        q = np.einsum("ni,ij,nj->n", diff, Sj_inv, diff)
        # full log-likelihood up to constant: -0.5*q - 0.5*log|Sj|
        scores[:, j] = -0.5 * q - 0.5 * logdet
    idx = np.argmax(scores, axis=1)
    ypred = classes[idx].reshape((-1, 1))
    if ytest.size == 0 or (ytest == 0).all():
        acc = 0.0
    else:
        acc = np.mean(ypred.ravel() == ytest.ravel())
    return acc, ypred


# ----------------------------
# Regression
# ----------------------------
def learnOLERegression(X, y):
    """
    Ordinary Least Squares: w = (X^T X)^{-1} X^T y
    """
    X = np.asarray(X)
    y = _as_column(y)
    XtX = X.T @ X
    Xty = X.T @ y
    w = inv(XtX) @ Xty
    return w


def learnRidgeRegression(X, y, lambd):
    """
    Ridge: w = (X^T X + λ I)^(-1) X^T y
    (Note: intercept column should be included in X if desired;
    we regularize all weights as per assignment spec.)
    """
    X = np.asarray(X)
    y = _as_column(y)
    d = X.shape[1]
    XtX = X.T @ X
    regI = lambd * np.eye(d)
    w = inv(XtX + regI) @ (X.T @ y)
    return w


def testOLERegression(w, Xtest, ytest):
    """
    Returns MSE on test set.
    """
    Xtest = np.asarray(Xtest)
    ytest = _as_column(ytest)
    ypred = Xtest @ w
    mse = np.mean((ytest - ypred) ** 2)
    return float(mse)


def regressionObjVal(w, X, y, lambd):
    """
    Objective for ridge regression (with 1/2 factors per spec):
       J(w) = 0.5 * ||y - Xw||^2 + 0.5 * λ ||w||^2
    Returns (error_scalar, gradient_vector)
    """
    X = np.asarray(X)
    y = _as_column(y)
    w = _as_column(w)
    resid = y - X @ w
    error = 0.5 * float(resid.T @ resid) + 0.5 * float(lambd * (w.T @ w))
    # gradient: -X^T(y - Xw) + λ w
    grad = -(X.T @ resid) + lambd * w
    return error, grad.ravel()


def mapNonLinear(x, p):
    """
    Map a single column vector x (N x 1) to polynomial features up to degree p:
        [1, x, x^2, ..., x^p]
    If p == 0 -> column of ones.
    """
    x = _as_column(x).ravel()
    N = x.shape[0]
    Xp = np.ones((N, p + 1))
    for deg in range(1, p + 1):
        Xp[:, deg] = x**deg
    return Xp


# ----------------------------
# Main script
# ----------------------------

# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open("sample.pickle", "rb"))
else:
    X, y, Xtest, ytest = pickle.load(open("sample.pickle", "rb"), encoding="latin1")

# LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print("LDA Accuracy = " + str(ldaacc))

# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print("QDA Accuracy = " + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title("LDA")

plt.subplot(1, 2, 2)
zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
plt.title("QDA")
plt.show()

if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open("diabetes.pickle", "rb"))
else:
    X, y, Xtest, ytest = pickle.load(open("diabetes.pickle", "rb"), encoding="latin1")

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print("MSE without intercept " + str(mle))
print("MSE with intercept " + str(mle_i))

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title("MSE for Train Data")
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title("MSE for Test Data")
plt.show()

k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {"maxiter": 200}  # a few more iterations helps CG converge

w_init = np.ones(X_i.shape[1])

for lambd in lambdas:
    args = (X_i, y, lambd)
    res = minimize(
        regressionObjVal, w_init, jac=True, args=args, method="CG", options=opts
    )
    w_l = res.x.reshape(-1, 1)

    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i += 1

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title("MSE for Train Data")
plt.legend(["Using scipy.minimize", "Direct minimization"])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title("MSE for Test Data")
plt.legend(["Using scipy.minimize", "Direct minimization"])
plt.show()


pmax = 7
lambda_opt = float(lambdas[np.argmin(mses3.ravel())])
print("lambda_opt (from Problem 3) =", lambda_opt)

mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    # No regularization
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    # With lambda_opt
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title("MSE for Train Data")
plt.legend(("No Regularization", "Regularization"))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title("MSE for Test Data")
plt.legend(("No Regularization", "Regularization"))
plt.show()
