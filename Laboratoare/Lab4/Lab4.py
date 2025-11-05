import numpy as np
import matplotlib.pyplot as plt
import timeit as tm
from scipy.io import wavfile



def exercitiul1():
    # calcul DFT (din Lab3) 
    def DFT(x):
        X = []
        N = len(x)
        for m in range(N):
            s = 0
            for k in range(N):
                s += x[k] * np.exp(-2j * np.pi * m * k / N)
            X.append(s)
        X = np.array(X)
        return X


    # calcul FFT 
    def FFT(x):
        # caz de baza
        N = len(x)
        if N == 1:
            return x
        
        # impartire dupa paritatea indicilor
        par = FFT(x[0::2])
        impar = FFT(x[1::2])

        # calcul, clasic dupa formula 
        X = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            f = np.exp(-2j * np.pi * k / N)
            X[k] = par[k] + f * impar[k]
            X[k + N // 2] = par[k] - f * impar[k]

        return X


    # dimensiunile vectorilor si timpii de calcul pentru cele 3 "metode"
    N_list = [128, 256, 512, 1024, 2048, 4096, 8192]
    time_DFT =[]
    time_FFT = []
    time_numpy = []

    for N in N_list:
        # vector random de dim N cu valori in [0, 1)
        x = np.random.rand(N)

        # DFT - manual
        start = tm.default_timer()
        DFT(x)
        end = tm.default_timer()
        time_DFT.append(end - start)

        # FFT - manual
        start = tm.default_timer()
        FFT(x)
        end = tm.default_timer()
        time_FFT.append(end - start)

        # FFT - numpy
        start = tm.default_timer()
        np.fft.fft(x)
        end = tm.default_timer()
        time_numpy.append(end - start)

    # afisare pe sc log
    plt.plot(N_list, time_DFT, label='DFT manual')
    plt.plot(N_list, time_FFT, label='FFT manual')
    plt.plot(N_list, time_numpy, label='FFT numpy')
    plt.yscale('log')
    plt.legend()
    plt.savefig("Images/ex1.pdf", format='pdf')
    plt.show()





def exercitiul2():
    fs = 20 # frecventa de esantionare
    f0, f1, f2 = 15, 35, 55 # frecventele semnalelor 15, 15+20 , 15+ 20*2
    t = np.linspace(0, 1, 1000, endpoint=False)
    n = np.linspace(0, 1, fs, endpoint=False)

    # semnalele continue 
    x = np.sin(2 * np.pi * f0 * t) 
    y = np.sin(2 * np.pi * f1 * t) 
    z = np.sin(2 * np.pi * f2 * t)

    # semnalele esantionate
    xd = np.sin(2 * np.pi * f0 * n) 
    yd = np.sin(2 * np.pi * f1 * n) 
    zd = np.sin(2 * np.pi * f2 * n)

    # afisare
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, x)
    axs[0].scatter(n, xd, color='orange')
    axs[1].plot(t, y)
    axs[1].scatter(n, yd, color='orange')
    axs[2].plot(t, z)
    axs[2].scatter(n, zd, color='orange')
    plt.savefig("Images/ex2.pdf", format='pdf')
    plt.show()





def exercitiul3(): 
    fs = 50 # noua frecv de esantionare
    f0, f1, f2 = 15, 65, 115 # frecventele semnalelor 
    t = np.linspace(0, 1, 1000, endpoint=False)
    n = np.linspace(0, 1, fs, endpoint=False) 

    # semnalele continue
    x = np.sin(2 * np.pi * f0 * t)
    y = np.sin(2 * np.pi * f1 * t)
    z = np.sin(2 * np.pi * f2 * t)

    # semnalele esantionate
    xd = np.sin(2 * np.pi * f0 * n)
    yd = np.sin(2 * np.pi * f1 * n)
    zd = np.sin(2 * np.pi * f2 * n)

    # afisare
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, x)
    axs[0].scatter(n, xd, color='purple')
    axs[1].plot(t, y)
    axs[1].scatter(n, yd, color='purple')
    axs[2].plot(t, z)
    axs[2].scatter(n, zd, color='purple')
    plt.savefig("Images/ex3.pdf", format='pdf')
    plt.show()






def exercitiul4():
    """
    f = 40 - 200 Hz => fmax = 200 Hz
    din Nyquist => fs >= 2 * fmax => fs >= 400 Hz
    """
    pass





def exercitiul5():
    """
    spectograma -> Images/ex5.pdf
    da, se poate distinge intre vocale
    a se distinge de restul vocalelor, prezinta cateva asemanari cu e si i
    i si e au forme foarte similare, dar i are cateva benzi luminoase in minus fata de e
    o si u se disting puternic de a, e, i prin benzile lor luminoase mai slabe | u are benzile mai intunecate decat o
    """
    pass





def exercitiul6():
    # pas a
    _, x = wavfile.read('Images/audio.wav')
    x = x.mean(axis=1) # stereo -> mono
    N = len(x)

    # pas b
    L = int(0.01 * N)
    pas = L // 2
    spectru = []

    # pas c
    for i in range(0, N - L, pas):
        segment = x[i:i+L]
        X = np.fft.fft(segment)
        spectru.append(np.abs(X[:L//2]))  

    # pas d 
    M = np.array(spectru).T

    # pas e
    # convertire + adaugare valoare mica 1e-6 pentru a evita log 0
    plt.imshow(10*np.log10(M + 1e-6), origin='lower', aspect='auto')
    plt.colorbar(label='db')
    plt.savefig("Images/ex6.pdf", format='pdf')
    plt.show()





def exercitiul7():
    """
    Notam A o variabila in unitate de masura absoluta si A(db) in decibeli.

    Ps(db) = 10 lg (Ps) 
    90 = 10 lg (Ps)
    Ps = 10^9

    SNR(db) = 10 lg (SNR)
    SNR(db) = 10 lg (Ps / Pz)
    80 = 10 lg (10^9 / Pz)
    8 = lg (10^9 / Pz)
    10^8 = 10^9 / Pz
    Pz = 10 (in valoarea absoluta, gen ca si putere)

    Pz(db) = 10 lg (Pz)
    Pz(db) = 10 lg (10) = 10 db (si in db)
    """
    pass





if __name__ == "__main__":
    # exercitiile 4, 5, 7 sunt rezolvare in comentariile de la functiile corespunzatoare
    # exercitiul1()
    # exercitiul2()
    exercitiul3()
    #exercitiul6()


