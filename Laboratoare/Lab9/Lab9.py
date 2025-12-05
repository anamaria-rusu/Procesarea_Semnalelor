import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
np.random.seed(21)


# a
# dim seriei
N = 1000
t = np.linspace(0, 1, N)

# componentele seriei
trend = t**2 + 0.04
sezon =  np.sin(2 * np.pi * 5 * t) +  np.sin(2 * np.pi * 7 * t)
var =  np.random.normal (0, 1, N) 

# seria de timp
st = trend + sezon + var


"""
pentru toate varinatele de exponential smoothing folosesc 
grid search + time-series cross-validation pentru a gasi cei mai ok params 
codul e cam la fel pt toate var, difera evident calculul lui s, paramatrii 
"""

# mediere exp 1.0
def exercitiul_b1():

    # calcul pentru exp smoothing (varianta simpla)
    def exp_sm_1(x,a):
        s = np.zeros(len(x))
        s[0] = x[0]
        for t in range(1,len(x)):
            s[t] = a*x[t] + (1-a)*s[t-1]
        return s


    best_a = None # cel mai ok alpha cautat
    best_mse = float('inf') # cel mai ok mse pentru comparatie 
    x = st.copy() # seria de timp notat cu x
    N = len(x) # lungimea seriei
    start = int(0.6*N) # punctul de start pentru cross-validation (aici 60% din date) ales dupa cateva incercari 

    # cautare grid pentru alpha intre 0.01 si 0.99
    for a in np.linspace(0.01, 0.99, 99):
        # erorile mse
        err = []

        # cross-validation
        for t in range(start, N):
            # primele t valori
            train = x[:t]
            # calculam predictia pentru pastul t
            pred = exp_sm_1(train, a)[-1]
            # calculam mse si add la vectorul de erori
            err.append((pred - x[t])**2)

        # o verificare 
        mse = np.mean(err) if len(err) > 0 else float('inf')

        # stabilire cel mai ok alpha  
        if mse < best_mse:
            best_mse = mse
            best_a = a

    # print + plot
    print(f"a : {best_a}")
    plt.plot(st)
    plt.plot(exp_sm_1(st, best_a), color='red')
    plt.savefig(f"Images/ex_b1.pdf", format='pdf')
    plt.show()





# mediere exp 2.0
def exercitiul_b2():

    def exp_sm_2(x, a, b):
        s  = np.zeros(len(x))   
        b_ = np.zeros(len(x))   
        s[0] = x[0]
        if len(x) > 1:
            b_[0] = x[1] - x[0]
        else:
            b_[0] = 0.0
        for t in range(1, len(x)):
            s[t]  = a * x[t] + (1 - a) * (s[t-1] + b_[t-1])
            b_[t] = b * (s[t] - s[t-1]) + (1 - b) * b_[t-1]
        return s, b_


    best_a = None
    best_b = None
    best_mse = float('inf')
    x = st.copy()
    N = len(x)
    start = int(0.6 * N)

    for a in np.linspace(0.01, 0.99, 20):
        for b in np.linspace(0.01, 0.99, 20):
            err = []
            for t in range(start, N):
                train = x[:t]
                s_train, b_train = exp_sm_2(train, a, b)
                pred = s_train[-1] + b_train[-1]  
                err.append((pred - x[t])**2)
            mse = np.mean(err) if len(err) > 0 else float('inf')

            if mse < best_mse:
                best_mse = mse
                best_a = a
                best_b = b

    print(f"a: {best_a}, b: {best_b}")
    s_opt, b_opt = exp_sm_2(st, best_a, best_b)
    plt.plot(st)
    plt.plot(s_opt+b_opt, color='red')
    plt.savefig("Images/ex_b2.pdf", format='pdf')
    plt.show()




# mediere exp 3.0
def exercitiul_b3():

    def exp_sm_3(x,a,b,g):
        L = 30
        s = np.zeros(len(x))
        b_ = np.zeros(len(x)) # b_ = b (din curs) si b = beta 
        c = np.zeros(len(x)) 
        s[0] = x[0]
        b_[0] = x[1] - x[0]

        for t in range(1, len(x)):
            if t - L >= 0: c_prev = c[t-L]
            else: c_prev = 0.0
            s[t] = a * (x[t] - c_prev) + (1 - a) * (s[t-1] + b_[t-1])
            b_[t] = b * (s[t] - s[t-1]) + (1 - b) * b_[t-1]
            if t - L >= 0:c[t] = g * (x[t] - s[t]) + (1 - g) * c_prev
            else:c[t] = 0.0

        return s, b_, c


    best_a = None
    best_b = None
    best_g = None
    best_mse = float('inf')
    x = st.copy()
    N = len(x)
    start = int(0.6*N)

    # am limitat la 5 pentru ca dura extrem de mult rularea 
    for a in np.linspace(0.01, 0.99, 5):
        for b in np.linspace(0.01, 0.99, 5):
            for g in np.linspace(0.01, 0.99, 5):
                err = []
                for t in range(start, N):
                    train = x[:t]
                    s_train, b_train, c_train = exp_sm_3(train, a, b, g)
                    pred = s_train[-1] + b_train[-1] + c_train[-1]
                    err.append((pred - x[t])**2)

                if len(err) == 0:
                    mse = float('inf')
                else:
                    mse = np.mean(err)
                    
                if mse < best_mse:
                    best_mse = mse
                    best_a = a
                    best_b = b
                    best_g = g


    print(f"a: {best_a}, b: {best_b}, g: {best_g}")
    s_opt, b_opt, c_opt = exp_sm_3(st, best_a, best_b, best_g)
    plt.plot(st)
    plt.plot(s_opt + b_opt + c_opt, color='red')
    plt.savefig(f"Images/ex_b3.pdf", format='pdf')
    plt.show()





def exercitiul_c(q=1):
    # seria
    x = st.copy()
    N = len(x)

    # datele
    x_train = x[:800]
    x_test  = x[800:]
    n = len(x_train)
    m = len(x_test)

    # media
    mu = np.mean(x_train)

    # deviatiile 
    epsilon_init = x_train - mu

    # estimare theta cu metoda celor mai mici patrate
    m_train = n - q
    Z = np.zeros((m_train, q))
    y_cent = np.zeros(m_train)

    row = 0
    for i in range(q, n):
        for k in range(1, q+1):
            Z[row, k-1] = epsilon_init[i - k]
        y_cent[row] = x_train[i] - mu
        row += 1
    theta = np.linalg.inv(Z.T @ Z) @ (Z.T @ y_cent)

    # reconstructie
    y_hat_train = np.full(n, np.nan)
    eps_hist = []

    for i in range(q):
        y_hat_train[i] = mu
        eps_i = x_train[i] - y_hat_train[i]
        eps_hist.append(eps_i)

    
    for i in range(q, n):
        last_q = np.array(eps_hist[-q:][::-1])  
        y_hat_train[i] = mu + last_q @ theta
        eps_i = x_train[i] - y_hat_train[i]
        eps_hist.append(eps_i)


    # prezicere
    y_hat_test = np.zeros(m)
    for t in range(m):
        last_q = np.array(eps_hist[-q:][::-1])
        y_hat = mu + last_q @ theta
        y_hat_test[t] = y_hat

        eps_new = x_test[t] - y_hat
        eps_hist.append(eps_new)

    
    plt.plot(x)
    plt.plot(y_hat_train, color ='red')
    plt.plot(range(n, N), y_hat_test, color='red')
    plt.savefig("Images/ex_c.pdf")
    plt.show()






def exercitiul_d():
    # seria
    P, Q = 20, 20
    x = st.copy()

    # datele
    x_train = x[:800]
    x_test  = x[800:]

    # model
    best_mse = float('inf')
    best_pq = None

    # modelul
    for p in range(1, P+1):
        for q in range(1, Q+1):
            model = ARIMA(x_train, order=(p, 0, q))
            model_fit = model.fit()
            pred_test = model_fit.forecast(steps=len(x_test))
            mse = np.mean((x_test - pred_test)**2)

            if mse < best_mse:
                best_mse = mse
                best_pq = (p, q)

    final_model = ARIMA(x, order=(best_pq[0], 0, best_pq[1])).fit()
    y_hat_full = final_model.fittedvalues 

    plt.figure(figsize=(10,4))
    plt.plot(x)
    plt.plot(y_hat_full)
    plt.savefig("Images/ex_d.pdf")
    plt.show()





if __name__ == "__main__":
    exercitiul_b1()
    exercitiul_b2()
    exercitiul_b3()
    exercitiul_c()
    exercitiul_d()
    pass