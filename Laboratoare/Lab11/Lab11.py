import numpy as np
import matplotlib.pyplot as plt



""" Ex 1 """
# dim seriei
N = 1000
t = np.linspace(0, 1, N)

# componentele seriei
trend = t**2 + 0.04
sezon =  np.sin(2 * np.pi * 5 * t) +  np.sin(2 * np.pi * 7 * t)
var =  np.random.normal (0, 1, N) 

# seria de timp
st = trend + sezon + var



""" Ex 2 """
# Calcul matrice Hankel pt serie
L = 510
X = np.zeros((L, N-L+1))
for i in range(L):
    for j in range(N-L+1):
        X[i, j] = st[i + j]
np.savetxt("Images/ex2.txt", X, fmt="%.4f")



""" Ex 3 """
# sursa: https://ee263.stanford.edu/lectures/svd.pdf
XXt = X @ X.T
XtX = X.T @ X

# desc in val proprii si vectori proprii
val_propr_XXt, vect_propr_XXt = np.linalg.eigh(XXt)
val_propr_XtX, vect_propr_XtX = np.linalg.eigh(XtX)

# descv svd
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# verificare (am verif doar pima comp)
r = len(S)
i = 0 
print("val  proprii XXt " + str(np.allclose(val_propr_XXt[::-1][:r], S**2, atol=1e-5)))
print("val  proprii XtX " + str(np.allclose(val_propr_XtX[::-1][:r], S**2, atol=1e-5)))
print("vect proprii XtX " + str(i) + " ", np.allclose(XtX @ Vt.T[:, i], S[i]**2 * Vt.T[:, i], atol=1e-5))
print("vect proprii XXt " + str(i) + " ", np.allclose(XXt @ U[:, i], S[i]**2 * U[:, i], atol=1e-5))



""" Ex 4 """
# Single Spectrum Analysis
# Hankel-izarea fiecarui Xi
def Hankel(Xi):
    L, K = Xi.shape
    x_hat = np.zeros(L + K - 1)
    count = np.zeros(L + K - 1)
    for i in range(L):
        for j in range(K):
            x_hat[i + j] += Xi[i, j]
            count[i + j] += 1
    x_hat /= count
    return x_hat


L = X.shape[0]
K = X.shape[1]
r = len(S)
x_hat = np.zeros((r, N))

for i in range(r):
    Xi = S[i] * np.outer(U[:, i], Vt[i, :])
    x_hat[i, :] = Hankel(Xi)
st_rec = x_hat.sum(axis=0)

print(" X = X_rec" + str(np.allclose(st, st_rec, atol=1e-5)))
plt.figure()
plt.plot(st, label="st")
plt.plot(st_rec, label="st_rec")
plt.legend()
plt.savefig(f"Images/ex4.pdf", format='pdf')
plt.show()