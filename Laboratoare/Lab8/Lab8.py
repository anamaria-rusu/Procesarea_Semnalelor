import numpy as np
import matplotlib.pyplot as plt



# dim seriei
N = 1000
t = np.linspace(0, 1, N)

# componentele seriei
trend = t**2 + 0.04
sezon =  np.sin(2 * np.pi * 5 * t) +  np.sin(2 * np.pi * 7 * t)
var =  np.random.normal (0, 1, N) 

# seria de timp
st = trend + sezon + var





def exercitiul_a():
    # afisarea comp si a seriei de timp
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(t, st)
    axs[0].set_title("seria de timp")
    axs[1].plot(t,trend)
    axs[1].set_title("trend")
    axs[2].plot(t, sezon)
    axs[2].set_title("sezon")
    axs[3].plot(t, var)
    axs[3].set_title("variatii")
    plt.tight_layout()
    plt.savefig(f"Images/ex1a.pdf", format='pdf')
    plt.show()





def exercitiul_b():
    # calcularea vectorului de autocorelatie
    st_ac = st.copy()
    st_ac = np.array(st_ac)
    st_ac = st_ac - np.mean(st_ac)
    ac = np.correlate(st_ac, st_ac, mode='full')
    plt.plot(ac)
    plt.savefig(f"Images/ex1b.pdf", format='pdf')
    plt.show()





def exercitiul_c(p = 100):
    st_c = st.copy()
    # pentru train consideram primele p si pentru test ultimile p
    train = st_c[:-p]      
    test  = st_c[-p:]      
    y = train[p:]
    m = len(y)           

    # calc matricea Y
    Y = np.zeros((m, p))
    for i in range(m):              
        for j in range(p):         
            Y[i, j] = train[i + p - j - 1]

    # calc x* (conform curs 8)
    GAMMA = Y.T @ Y
    gamma = Y.T @ y
    x_star = np.linalg.inv(GAMMA) @ gamma

    # predictiile 
    predictii = []
    history = train.copy()         

    for i in range(p):
        # luam ultimele p valori din history
        last_p = history[-p:]  
        # inversam ordinea     
        last_p = last_p[::-1]   
        # calc predictia    
        y_hat = x_star @ last_p
        # adaugam predictia in vectorul de predictii si in history
        predictii = np.append(predictii, y_hat)
        history = np.append(history, y_hat)

    # afisare
    t_pred = t[len(train):len(train) + len(predictii)]   
    plt.plot(t, st)        
    plt.plot(t_pred, predictii)
    plt.plot(t_pred, test)
    plt.savefig(f"Images/ex1c.pdf", format='pdf')
    plt.show()





def exercitiul_d():

    # cam la fel cu cel de mai sus de la c
    def AR(p,m):
        st_c = st.copy()
        N = len(st_c)

        if N <= m + p:
            return None  

        train = st_c[:-m]  
        test  = st_c[-m:]   
        N_train = len(train)

        if N_train <= p:
            return None

        y = train[p:]
        m_train = len(y)

        Y = np.zeros((m_train, p))
        for i in range(m_train):
            for j in range(p):
                Y[i, j] = train[i + p - j - 1]

        GAMMA = Y.T @ Y
        gamma = Y.T @ y
        try:
            x_star = np.linalg.inv(GAMMA) @ gamma
        except np.linalg.LinAlgError:
            return None
        
        predictii = np.zeros(m)
        history = train.copy()

        for i in range(m):
            last_p = history[-p:]     
            last_p = last_p[::-1]     
            y_hat = x_star @ last_p
            predictii[i] = y_hat
            history = np.append(history, y_hat)

        # estimez performanta de baza mse 
        mse = np.mean((predictii - test)**2)
        return mse


    best_p = None
    best_m = None
    best_mse = np.inf
    for p in range(5,50):
        for m in range(5,100):
            mse = AR(p, m)
            if mse is None:
                continue

            #print(f"p={p}, m={m}, MSE={mse}")
            if mse < best_mse:
                best_mse = mse
                best_p = p
                best_m = m
              
    print("p optim:", best_p)
    print("m optim:", best_m)



if __name__ == "__main__":
    exercitiul_a()
    exercitiul_b()
    exercitiul_c()
    exercitiul_d()




