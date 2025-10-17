"""
NOTES:

x(t) = A sin (2πft + φ) = A sin (ωt + φ) = semnal sinusoidal
cos (t) = sin (t + π/2)
sin (t) = cos (t - π/2)

discret: x[n] = A sin (2 π fn Ts + φ) 
fs = 1/Ts = frecventa de esantionare
discret -> periodic <=> N = 2πk / ω din Z, k din Z


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io, signal
import sounddevice as sd


def exercitiul1(A, f, phi):
    t = np.linspace(0,2,200)
    sin_ = A * np.sin(2 * np.pi * f * t + phi)
    cos_ = A *np.cos(2 * np.pi * f * t + phi - np.pi/2)

    fig, axs = plt.subplots(2)

    axs[0].plot(t,sin_)
    axs[0].set_title("sin")
    axs[0].grid()

    axs[1].plot(t,cos_)
    axs[1].set_title("cos")
    axs[1].grid()

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("Images/ex1.pdf", format='pdf')
    plt.show()





def exercitiul2(f, phiList):
    # daca nu s-a dat vreun phi, oprim executia 
    if not phiList:
        return None 

    t = np.linspace(0,2,200)
    fig, ax = plt.subplots(figsize=(10, 5))

    # trecem prin fiecare element din phi si desenam semnalul coresp
    for i, phi in enumerate(phiList):
        sin_ = np.sin(2 * np.pi * f * t + phi)       
        ax.plot(t, sin_)
    ax.grid()
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("Images/ex2.pdf", format='pdf')
    plt.show()


    # SNR = ||x|| ^ 2 / gamma^2 * ||z||^2
    # semnalul x este generat cu faza phiList[0]
    x = np.sin(2* np.pi * f * t + phiList[0])
    # semnalul z este zgomot gaussian (dist Gaussiana standard)
    z = np.random.normal(0,1,len(t))
   
    SNR = [0.1, 1, 10, 100]
    fig, ax = plt.subplots(len(SNR), 1, figsize=(8, 6))

    # penetru fiecare SNR aflam parametrul gamma si generam semnalul
    # pe subplot se afla semnalul original si cel cu zgomot
    for i, snr in enumerate(SNR):
        gamma = np.linalg.norm(x) / (np.sqrt(snr) * np.linalg.norm(z))
        xn = x + gamma * z
        ax[i].plot(t, x, label='original')
        ax[i].plot(t, xn, label='zgomot')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("Images/ex2_SNR.pdf", format='pdf')
    plt.show()





def exercitiul3():
    # semnale a-d din Laborator 1
    fs = 44100

    # subpunct a
    t = np.linspace(0,4,4*fs)
    a = np.sin(2 * np.pi * 400 * t)
    sd.play(a, fs)
    sd.wait()

    # subpunct b
    t = np.linspace(0,3, 3*fs)
    b = np.sin(2 * np.pi * 800 * t)
    sd.play(b, fs)
    sd.wait()

    # subpunct c
    t = np.linspace(0, 1, fs)
    c = 240 * t - np.floor(240 * t)
    sd.play(c, fs)
    sd.wait()

    # subpunct d
    t = np.linspace(0,5, 5*fs)
    d = np.sign(np.sin(2 * np.pi * 300 * t))
    sd.play(d, fs)
    sd.wait()

    # salvare semnal d .wav
    io.wavfile.write("Images/ex3.wav", fs, d)
    fs_read, data_read = io.wavfile.read("Images/ex3.wav")

    # incarcare de pe disk (functioneaza)
    sd.play(data_read, fs_read)
    sd.wait()





def exercitiul4():
    t = np.linspace (0, 1, 100)
    # semnalul sinusoidal de 400 Hz
    sin_  = np.sin(2 * np.pi * 500* t)
    # semnalul sawtooth de 250 Hz
    saw_ = signal.sawtooth(2 * np.pi * 250 * t)
    # suma celor doua semnale
    sum_ = sin_ + saw_

    # plotarea celor trei semnale
    fig, axs = plt.subplots(3)
    axs[0].plot(t,sin_)
    axs[0].set_title("sin")

    axs[1].plot(t,saw_)
    axs[1].set_title("sawtooth")

    axs[2].plot(t,sum_)
    axs[2].set_title("sum")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("Images/ex4.pdf", format='pdf')
    plt.show()





def exercitiul5():
    fs = 44100
    t = np.linspace(0,5,5*fs)
    # doua semnale cosinusoidale cu f de 300 si de 600 (Hz)
    x = np.cos(2 * np.pi * 300 * t)
    y = np.cos(2 * np.pi * 600 * t)
    z = np.concatenate((x, y))
  
    # redare semnal concatenat
    sd.play(z, fs)
    sd.wait()
    io.wavfile.write("Images/ex5.wav", fs, z)

    """
    OBS: pentru exemplul ales, primul semnal (adica x) se aude la o frecventa mai joasa
    in comparatie cu al doilea semnal y care se aude la o frecventa mai inalta
    """



    

def exercitiul6():
    fs = 4000
    t = np.linspace(0, 8, fs)
    x = np.sin(2 * np.pi * (fs/2) * t)
    y = np.sin(2 * np.pi * (fs/4) * t)
    z = np.sin(0 * t)

    fig, axs = plt.subplots(3)
    axs[0].plot(t,x)
    axs[0].set_title("x")

    axs[1].plot(t,y)
    axs[1].set_title("y")

    axs[2].plot(t,z)
    axs[2].set_title("z")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("Images/ex6.pdf", format='pdf')
    plt.show()

    """
    OBS: (fs = 4000)
    - pentru f = fs/2 = 2000 Hz avem o frecventa mai mare (adica semnalul se repeta mai des - apar mai multe "valuri")
    - pentru f = fs/4 = 1000 Hz avem o frecventa mai mica (adica semnalul se repeta mai rar prin comparatie cu primul - apar mai putine "valuri") comparativ cu f//2
    - pentru f = 0 Hz.. e doar 0 (adica semnalul nu se schimba in timp)
    """





def exercitiul7():
    def rezolavre(fs):
        # am ales f = 1, deci nu mai apare direct in ecuatie
        # semnal initial
        t1 = np.linspace(0, 1, fs)
        x = np.sin(2 * np.pi * t1)
        # semnal decimat la 1/4 incepand din pozitia 0
        t2 = t1[0::4]
        y = np.sin(2 * np.pi * t2)
        # semnal decimat la 1/4 incepand din pozitia 1
        t3 = t1[1::4]
        z = np.sin(2 * np.pi * t3 +  2*np.pi*(1/fs))

        fig, axs = plt.subplots(3,1, figsize=(8, 8))
        axs[0].stem(t1,x)
        axs[0].set_title("original")

        axs[1].stem(t2,y)
        axs[1].set_title("decimat din poz 0")

        axs[2].stem(t3,z)
        axs[2].set_title("decimat din poz 1")

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig("Images/ex7_" + str(fs) + ".pdf", format='pdf')
        plt.show()

    rezolavre(100)
    rezolavre(1000)

    """
    OBS:
    PUNCTUL A:
    - forma generala ramane la fel (adica tot o forma sinusoidala),
      doar ca x (semnalul original) are mai multe esantioane decat y 

    PUNCTUL B:
    - pentru fs = 1000 nu se distingeau intocmai diferentele dintre y si z
    - pentru fs = 100 se observau mai clar deci,
        - cele doua au aceeasi frecventa 
        - dar difera faza (phi) dintre cele doua semnale
    """





def exercitiul8():
    # spatiul de la -pi/2 la pi/2
    t = np.linspace(-np.pi/2,np.pi/2,300)

    # sinusul real
    sin_ = np.sin(t)
    # aproximarea Taylor  sin(t) = t
    taylor = t
    # aproximarea Pade sint(t) = pade(t)
    pade = (t - (7.0/60.0) * t**3) / (1.0 + (1.0/20.0) * t**2)

    # plotarea celor trei semnale
    # diferenta dintre sin si pade este extrem de mica (vezi Lab2\Images\SinPade.png)
    plt.figure(figsize=(10, 5))
    plt.plot(t,sin_, label='sinus')
    plt.plot(t, taylor, label='Taylor')
    plt.plot(t, pade, label='Pade')
    plt.legend()
    plt.grid()
    plt.savefig("Images/ex8_SinTaylorPade.pdf", format='pdf')
    plt.show()

    # calculul erorilor + plot
    errT = sin_ - taylor 
    errP = sin_ - pade

    # afisare pe axa Oy liniara
    plt.figure(figsize=(10, 5))
    plt.plot(t, errT, label='Taylor')
    plt.plot(t, errP, label='Pade')
    plt.grid()
    plt.legend()
    plt.savefig("Images/ex8_EroareLiniar.pdf", format='pdf')
    plt.show()

    # afisare pe axa Oy logaritmica 
    plt.figure(figsize=(10,5))
    plt.plot(t, np.abs(errT), label="Taylor")
    plt.plot(t, np.abs(errP),   label="Pade")
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig("Images/ex8_EroareLog.pdf", format='pdf')
    plt.show()







if __name__ == "__main__":
    while True:
        x = int(input("Ex:"))
        if x == 1:
            exercitiul1(1, 2, np.pi/4) 
        elif x == 2:
            exercitiul2(4, [np.pi/4, np.pi/2,np.pi/3,np.pi/31])
        elif x == 3:
            exercitiul3()
        elif x == 4:
            exercitiul4()
        elif x == 5:
            exercitiul5()
        elif x == 6:
            exercitiul6()
        elif x == 7:
            exercitiul7()
        elif x == 8:
            exercitiul8()
        else:
            break

