import numpy as np
import matplotlib.pyplot as plt
from l1regls import l1regls
import cvxopt   

""" a """
# dim seriei
N = 1000
t = np.linspace(0, 1, N)

# componentele seriei
trend = t**2 + 0.04
sezon =  np.sin(2 * np.pi * 5 * t) +  np.sin(2 * np.pi * 7 * t)
var =  np.random.normal (0, 1, N) 

# seria de timp
st = trend + sezon + var





""" b """ 
# Modelul Ar cu orizon p
p = 100
st_copy = st.copy()
y = st_copy[p:]
m = len(y)

Y = np.zeros((m, p))
for i in range(m):
    for j in range(p):
        Y[i, j] = st_copy[i + p - j - 1]
x_star = np.linalg.inv(Y.T @ Y) @ (Y.T @ y)





""" c """

""" GREEDY """
# pp sa selectam k cele mai bune lag-uri 
k = 10
m, p = Y.shape
# normalizare
norm_cols = np.linalg.norm(Y, axis=0)
Y_greedy = Y / norm_cols

# lagurile alese 
sol = []
r = y.copy()

# alegem cele mai bune k laguri 
for i in range(k):
    # lag-ul cel mai corealt 
    cor = np.abs(Y_greedy.T @ r)
    # pentru a nu alege de doua ori acelasi lag
    cor[sol] = -1 
    best_lag = int(np.argmax(np.abs(cor)))
    sol.append(best_lag)

    # recalculare coficienti
    A = Y_greedy[:, sol]
    x_star_greedy = np.linalg.inv(A.T @ A) @ (A.T @ y)
    r = y - A @ x_star_greedy

x_sparse = np.zeros(p)
x_sparse[sol] = x_star_greedy/ norm_cols[sol]



""" REGRESIE """
# regresia L1 (dpa exemplul din link)
lambda_ = 10
scale = 1/np.sqrt(lambda_)
A = cvxopt.matrix(scale * Y)
b = cvxopt.matrix((scale * y).reshape(-1,1))
x_l1 = np.array(l1regls(A,b)).flatten()





""" d """
def calcul_coeficienti(c):
    # matricea companion
    c = np.asarray(c)
    C = np.zeros((len(c), len(c)))
    C[1:, :-1] = np.eye(len(c) - 1)
    C[:, -1] = -c
    # val proprii q -> det(C - q*I) = 0
    q = np.linalg.eigvals(C)
    return q
    



""" e """
# x statioana <=> daca radacinile sunt in afara cerului unitate, adica abs < 1
# obtin True pentru toate
print("orginal: " + str(np.all(np.abs(calcul_coeficienti(x_star)) < 1)))
print("greedy: " + str(np.all(np.abs(calcul_coeficienti(x_sparse)) < 1)))
print("l1: " + str(np.all(np.abs(calcul_coeficienti(x_l1)) < 1)))