import numpy as np
import matplotlib.pyplot as plt
import scipy as sp



def exercitiul1(B = 1):

    # semnalul 
    t = np.linspace(-3, 3, 1000)
    x = np.sinc(B*t) ** 2

    # frecventele de esantionare incercate 
    FS = [1, 1.5, 2, 4]
    fig, axs = plt.subplots(len(FS), 1)

    # reconstruirea semnalului pentru fiecare fs
    for i, fs in enumerate(FS):
        Ts = 1 / fs
        tn = np.arange(-3, 3, Ts)
        xn = np.sinc(B * tn) ** 2

        # reconstructia semnalului
        xhat = sum(xn[k] * np.sinc((t - tn[k]) / Ts) for k in range(len(tn)))

        axs[i].plot(t, x)
        axs[i].plot(t, xhat, '--')
        axs[i].stem(tn,xn)

    plt.savefig("Images/ex1.pdf", format='pdf')
    plt.show()





def exercitiul2(N = 100):

    # a. pentru semnal random
    x = np.random.rand(N)

    fig, axs = plt.subplots(4, 1)
    for ax in axs:
        ax.plot(x)
        x = np.convolve(x,x)

    # OBS: semnalul devinde din ce in ce mai "neted"
    plt.savefig("Images/ex2a.pdf", format='pdf')
    plt.show()


    # b. pentru semnal bloc rectangular
    # am luat y = 1 in intervalul [N/3, 2N/3), 0 in rest
    y = np.zeros(N)
    y[N//3: 2*N//3] = 1 

    fig, axs = plt.subplots(4, 1)
    for ax in axs:
        ax.plot(y)
        y = np.convolve(y,y)

    # OBS: initial avem un dreptughi (aproximativ), apoi un triunghi, si apoi o forma parabolica din ce in ce mai "ascutita" (adica cu "baza" din ce in ce mai ingusta)
    plt.savefig("Images/ex2b.pdf", format='pdf') 
    plt.show()





def exercitiul3(N = 4):

    # functie pt printrea polindom 
    def print_polinom(coefs):
        return " + ".join(f"{coef}x^{i}" if i > 0 else f"{coef}" for i, coef in enumerate(coefs))


    # generarepolinoame p & q
    # am luat coef intre -16 si 15
    p = [np.random.randint(-2**4, 2**4) for _ in range(N+1)]
    q = [np.random.randint(-2**4, 2**4) for _ in range(N+1)]

    print("p = ", print_polinom(p))
    print("q = ", print_polinom(q))


    # inmultire clasica O(N^2)
    r_classic = [0] * (2*N+1)
    for i in range(N+1):
        for j in range(N+1):
            r_classic[i+j] += p[i] * q[j]

    print("r_classic = ", print_polinom(r_classic))


    # inmultire fft O(N log N)
    p_fft = np.fft.fft(p,2*N+1)
    q_fft = np.fft.fft(q,2*N+1)
    r_fft = np.fft.ifft(p_fft * q_fft).real.round().astype(int)

    print("r_fft     = ", print_polinom(r_fft))





def exercitiul4(n=20, d=3):

    # vectorii x & y (care are practic o schimbare de faza)
    d = d % n 
    x = np.random.rand(n)
    y = np.roll(x, d)

    c1 = np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y))
    c1_real = np.real(c1)
    c1_index = np.argmax(c1_real)

    c2 = np.fft.ifft(np.fft.fft(y) / np.fft.fft(x))
    c2_real = np.real(c2)
    c2_index = np.argmax(c2_real)

    print("d = ", d)
    print("d1 = ", c1_index)
    print("d2 = ", c2_index)

    # ploatre pt obs
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(x, marker='o')
    axs[0].plot(y, marker='x')
    axs[1].plot(c1_real, marker='o')
    axs[1].axvline(c1_index, color='r', linestyle='--')
    axs[2].plot(c2_real, marker='o')
    axs[2].axvline(c2_index, color='r', linestyle='--')
    plt.savefig("Images/ex4.pdf", format='pdf') 
    plt.show()






def exercitiul5():

    # semnalul
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 100 * t)
    Nw = 200

    # fereastra dreptunchiular 
    dr_window = np.ones(Nw)
    dr_sin = x[:Nw] * dr_window

    # fereastra Hann
    hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(Nw) / Nw))
    hann_sin = x[:Nw]  * hann_window

    # afisare
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t[:Nw], dr_sin)
    axs[1].plot(t[:Nw], hann_sin)
    plt.savefig("Images/ex5.pdf", format='pdf')
    plt.show()





def exercitiul6():

    # a 
    # incarcare date 
    # primele 72 de inreg --> 3 zile 
    train = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=str)
    x = train[:72,2].astype(float)



    # b
    ws = [5,9,13,17]
    plt.plot(x, label='og')
    for w in ws:
        fma = np.convolve(x, np.ones(w), 'valid') / w
        plt.plot(fma, label=f'w={w}')
    plt.savefig("Images/ex6b.pdf", format='pdf')
    plt.show()



    # c 
    # f de taiere = f maxima pe care filtrul o mai lasa sa treaca fara atenuare seminificativa
    fs = 1 / 3600
    f_cut = 1 / (12 * 3600)
    f_nyquist = fs / 2
    wn = f_cut / f_nyquist
    print("f_cut =", f_cut)
    print("w_cut =", wn)



    # d
    N = 5   
    rp = 5

    # proiectare filtre
    b_butter, a_butter = sp.signal.butter(N=N, Wn=wn, btype='low')
    b_cheby, a_cheby = sp.signal.cheby1(N=N, rp=rp, Wn=wn, btype='low')

    # raspuns in frecventa filtre
    wa, ha = sp.signal.freqz(b_butter, a_butter)
    wb, hb = sp.signal.freqz(b_cheby, a_cheby)

    # pt err la runtime pt log(0)
    epsilon = 1e-15

    # afisare
    plt.plot(wa, 20 * np.log10(np.abs(ha) + epsilon), label="butterworth")
    plt.plot(wb, 20 * np.log10(np.abs(hb) + epsilon), label="chebyshev")
    plt.savefig("Images/ex6d.pdf", format='pdf')
    plt.show()



    # e - mai potrivit e Chebyshev pentru ca urmareste mai fidel semnalul original (prin comp cu Butterworth)
    x_filt_butter = sp.signal.filtfilt(b_butter, a_butter, x)
    x_filt_cheby  = sp.signal.filtfilt(b_cheby, a_cheby, x)

    plt.plot(x, label="original",)
    plt.plot(x_filt_butter, label="butterworth")
    plt.plot(x_filt_cheby, label="chebyshev")
    plt.legend()
    plt.savefig("Images/ex6e.pdf", format='pdf')
    plt.show()



    # f
    Ns = [1, 13]
    rps = [2, 14]

    for N in Ns:
        # butterworth
        b_butter, a_butter = sp.signal.butter(N=N, Wn=wn, btype='low')
        x_butter = sp.signal.filtfilt(b_butter, a_butter, x)
        plt.plot(x_butter, label=f'butterworth N={N}')

        # chebyshev
        for rp in rps:
            b_cheby, a_cheby = sp.signal.cheby1(N=N, rp=rp, Wn=wn, btype='low')
            x_cheby  = sp.signal.filtfilt(b_cheby, a_cheby, x)
            plt.plot(x_cheby, label=f'chebyshev N={N}, rp={rp}', color='red')

        plt.plot(x, label="original", color='black')
        plt.legend()
        plt.savefig(f'Images/ex6f-N={N}.pdf', format='pdf')
        plt.show()





if __name__ == "__main__":
    exercitiul1()
    exercitiul2()
    exercitiul3()
    exercitiul4()
    exercitiul5()
    exercitiul6()
