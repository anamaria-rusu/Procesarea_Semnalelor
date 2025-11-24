from scipy import datasets
import numpy as np
import matplotlib.pyplot as plt



def exercitiul1(N = 128):

    # pentru afisare 
    def show(image, freq_db, ex):
        fix, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(freq_db)
        plt.savefig(f"Images/ex1_{ex}.pdf", format='pdf')
        plt.show()



    # a 
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    n1, n2 = np.meshgrid(x, y)
    # imagine
    image = np.sin(2*np.pi*n1 + 3*np.pi*n2)
    # spectru 
    Y = np.fft.fft2(image)
    freq_db = 20*np.log10(np.abs(Y) + 1e-6)    
    show(image, freq_db, "a")

   

    # b
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    n1, n2 = np.meshgrid(x, y)
    # imagine 
    image = np.sin(4*np.pi*n1) + np.cos(6*np.pi*n2)
    # spectru
    Y = np.fft.fft2(image)
    freq_db = 20*np.log10(np.abs(Y) + 1e-6)
    show(image, freq_db, "b")



    #c 
    Y = np.zeros((N, N), dtype=complex)
    Y[0, 5] = 1
    Y[0, N-5] = 1
    # imaginea
    x = np.fft.ifft2(Y).real
    # spectru
    freq_db = 20 * np.log10(np.abs(Y) + 1e-6)
    show(x, freq_db, "c")



    # d 
    Y = np.zeros((N, N), dtype=complex)
    Y[5, 0] = 1
    Y[N-5, 0] = 1
    # imaginea
    x = np.fft.ifft2(Y).real
    # spectru    
    freq_db = 20 * np.log10(np.abs(Y) + 1e-6)
    show(x, freq_db, "d")



    # e
    Y = np.zeros((N, N), dtype=complex)
    Y[5, 5] = 1
    Y[N-5, N-5] = 1
    # imaginea
    x = np.fft.ifft2(Y).real
    # spectru
    freq_db = 20 * np.log10(np.abs(Y) + 1e-6)
    show(x, freq_db, "e")




# functie pentru calcul SNR
def snr_(original, processed):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)
    noise = original - processed
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return np.inf 
    return 20 * np.log10(signal_power / noise_power + 1e-6)





def exercitiul2(X, strat = 0, end = 200, step = 10, SNR=0.5, ex3 = False):

    # transformam in dom de frecventa
    Y = np.fft.fft2(X)

    # spectru (+ un offset mic pentru a evita log(0) - err la runtime)
    freq_db = 20 * np.log10(np.abs(Y) + 1e-6)

    # luam un range de frecvente 
    # comprimam imaginea pana cand ajungem la SNR dorit
    for freq_cutoff in range(strat, end, step):
        Y_cutoff = Y.copy()
        Y_cutoff[freq_db > freq_cutoff] = 0
        X_cutoff = np.fft.ifft2(Y_cutoff)
        X_cutoff = np.real(X_cutoff)
        if snr_(X, X_cutoff) > SNR:
            break

    # save + afisare
    if not ex3:
        plt.imshow(X_cutoff, cmap=plt.cm.gray)
        plt.savefig(f"Images/ex2.pdf", format='pdf')
        plt.show()
    return X_cutoff
       

    


def exercitiul3(X):

    # raton cu noise din Lab7
    freq_cutoff = 120  
    pixel_noise = 200
    noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
    X_noisy = X + noise
    snr_before = snr_(X, X_noisy)
    print("snr before =", snr_before)

    # am apelat functia de ela ex2 care va comprima imaginea
    # considerand freq_cutoff = 120 (for-ul av face un singur pas) 
    # eliminare sgomot -> atenuare frecvente inalte 
    X_denoised = exercitiul2(X_noisy, freq_cutoff, freq_cutoff+1, 1, SNR=0.5, ex3=True)
    snr_after = snr_(X, X_denoised)
    print("snr after =", snr_after)

    # save + afisare
    fix, axs = plt.subplots(1, 3)
    axs[0].set_title("Raton Original")
    axs[0].imshow(X, cmap=plt.cm.gray)
    axs[1].set_title("Raton Zgomotos")
    axs[1].imshow(X_noisy, cmap=plt.cm.gray)
    axs[2].set_title("Raton Curatat")
    axs[2].imshow(X_denoised, cmap=plt.cm.gray)
    plt.savefig(f"Images/ex3.pdf", format='pdf')
    plt.show()

  



if __name__ == "__main__":
    X = datasets.face(gray=True)
    exercitiul1()
    exercitiul2(X)
    exercitiul3(X)