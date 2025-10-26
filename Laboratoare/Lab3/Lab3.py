from pickletools import dis
import numpy as np
import matplotlib.pyplot as plt


def exercitiul1(N=8):

    # generarea matricei Fourier de dim N*N
    F = np.zeros((N,N), dtype=complex)
    for ix in range(N):
        for jx in range(N):
            F[ix,jx] = np.exp(-2* np.pi * 1j * ix * jx / N)
    # salvare matrice intr-un .txt
    np.savetxt("Images/ex1.txt", F)

    # afisarea in subploturi a partii re si im pt fiecarea linie din F
    fig, axs = plt.subplots(N, 1, figsize=(N, 4*N))
    for ix in range(N):
        axs[ix].plot(F[ix].real, label='Re')
        axs[ix].plot(F[ix].imag, label='Im')
    plt.tight_layout()
    plt.savefig("Images/ex1.pdf", format='pdf')
    plt.show()

    # verificare daca F e unitara
    # FH = transpusa conjugata a lui F
    # np.eye(N) -  matricea identitate de dim N*N 
    # salvare rez verif in acelasi file 
    FH = np.conj(np.transpose(F))
    res = np.allclose(FH @ F, N * np.eye(N), atol=1e-5)
    with open("Images/ex1.txt", "a") as f:
        f.write("\nF unitara : " + str(res) + "\n")





def exercitiul2():
    f = 7    # frecventa 
    i = 404  # esantion : 404
    t = np.linspace(0,1,1000)
    x = np.sin(2 * np.pi * f * t)
    d  = np.abs(x) 

    # FIGURA 1 - cu culori ?
    # semnal sinusoidal x
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].scatter(t, x, c=d, s=5)
    axs[0].scatter(t[i], x[i]) # esantion i

    # infasurarea semnalului x in planul complex
    y = x * np.exp(-2j * np.pi * t)
    axs[1].scatter(y.real, y.imag, c=d, s=5) 
    axs[1].set_aspect('equal', adjustable='box')  # pentru aspect 
    axs[1].scatter(t[i], x[i]) # esantion i
    axs[1].axhline(0, color='black', linewidth=1) # axa x
    axs[1].axvline(0, color='black', linewidth=1) # axa y
    plt.savefig("Images/ex2Fig1.pdf", format='pdf')
    plt.show()



    # FIGURA 2 - fara culori
    O = [1,2,5,7]
    fig, axs2 = plt.subplots(1,4)
    for i, o in enumerate(O):
        z = x * np.exp(-2j * np.pi * t * o)
        axs2[i].plot(z.real, z.imag)
        axs2[i].axhline(0, color='black', linewidth=1) # axa x
        axs2[i].axvline(0, color='black', linewidth=1) # axa y
        axs2[i].set_aspect('equal', adjustable='box')  # pentru aspect
    plt.tight_layout()
    plt.savefig("Images/ex2Fig2.pdf", format='pdf')
    plt.show()





def exercitiul3():
    # semnalul compus din 3 sin de f 31-47-113 Hz
    t = np.linspace(0,1,1000)
    x = np.sin(2*np.pi*31*t) + np.sin(2*np.pi*47*t) + np.sin(2*np.pi*113*t)
    plt.plot(t, x)
    plt.savefig("Images/ex3Fig1.pdf", format='pdf')
    plt.show()

    # calcul dtf dupa Relatia 1
    n = len(t)
    X = []  
    for m in range(n):
        s = 0
        for k in range(n):
            s += x[k] * np.exp(-2j * np.pi * k * m / n)
        X.append(s)
    X = np.array(X)
    modX = np.abs(X)

    fs = 1 / (t[1] - t[0]) # frecv de esantionare
    fa = np.linspace(0, fs, n) # frecventele analizate <=> fa = [m * fs / n for m in range(n)]

    plt.stem(fa[:n//2], modX[:n//2])
    plt.savefig("Images/ex3Fig2.pdf", format='pdf')
    plt.show()





if __name__ == "__main__":
    exercitiul1()
    exercitiul2()
    exercitiul3()