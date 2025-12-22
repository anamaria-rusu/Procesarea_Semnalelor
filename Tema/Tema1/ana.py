import numpy as np
import matplotlib.pyplot as plt
import cv2 
from scipy.fft import dctn, idctn
from scipy import datasets


# matricea de cuantizare standard (pentru imaginile monocrome)
Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

Q_jpeg_chrominance = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


# conversie RGB -> YCbCr 
def rgb_to_ycbcr(X):

    X_ycbcr = np.zeros_like(X, dtype=np.float32)
    h, w, _ = X.shape

    for i in range(h):
        for j in range(w):

            R = float(X[i, j, 0])
            G = float(X[i, j, 1])
            B = float(X[i, j, 2])

            Y  = 16  + 0.257 * R + 0.504 * G + 0.098 * B
            Cb = 128 - 0.148 * R - 0.291 * G + 0.439 * B
            Cr = 128 + 0.439 * R - 0.368 * G - 0.071 * B

            X_ycbcr[i, j, 0] = np.clip(Y,  16, 235)
            X_ycbcr[i, j, 1] = np.clip(Cb, 16, 240)
            X_ycbcr[i, j, 2] = np.clip(Cr, 16, 240)

    return X_ycbcr



# conversie YCbCr -> RGB 
def ycbcr_to_rgb(X_compressed):

    X_rgb = np.zeros_like(X_compressed, dtype=np.float32)
    h, w, _ = X_compressed.shape

    for i in range(h):
        for j in range(w):

            Y  = float(X_compressed[i, j, 0]) - 16.0
            Cb = float(X_compressed[i, j, 1]) - 128.0
            Cr = float(X_compressed[i, j, 2]) - 128.0

            R = 1.164 * Y + 1.596 * Cr
            G = 1.164 * Y - 0.392 * Cb - 0.813 * Cr
            B = 1.164 * Y + 2.017 * Cb

            X_rgb[i, j, 0] = np.clip(R, 0, 255)
            X_rgb[i, j, 1] = np.clip(G, 0, 255)
            X_rgb[i, j, 2] = np.clip(B, 0, 255)

    return X_rgb



""" Cerinta 1 """
def compress_monochrome_image(X, s = 1):
        
    Q = s * Q_jpeg
    h, w = X.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    X = X[:h, :w]  

    X_compressed = np.zeros_like(X, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):

            # se ia fiecare pachet de 8x8
            x = X[i:i+8, j:j+8]
            y = dctn(x)
            y_jpeg = Q * np.round(y / Q)
            x_jpeg = idctn(y_jpeg)
            # se insereaza inapoi in imaginea comprimata
            X_compressed[i:i+8, j:j+8] = x_jpeg

    return X_compressed



""" Cerinta 2 """
def compress_color_image(X, s=1):
    
    QY = s * Q_jpeg
    QC = s * Q_jpeg_chrominance
    h, w, _ = X.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    X = X[:h, :w]

    X_ycbcr = rgb_to_ycbcr(X).astype(np.float32)
    X_compressed = np.zeros_like(X_ycbcr, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # Y
            x = X_ycbcr[i:i+8, j:j+8, 0]
            y = dctn(x)
            y_jpeg = QY * np.round(y / QY)
            X_compressed[i:i+8, j:j+8, 0] = idctn(y_jpeg)

            # Cb
            x = X_ycbcr[i:i+8, j:j+8, 1]
            y = dctn(x)
            y_jpeg = QC * np.round(y / QC)
            X_compressed[i:i+8, j:j+8, 1] = idctn(y_jpeg)

            # Cr
            x = X_ycbcr[i:i+8, j:j+8, 2]
            y = dctn(x)
            y_jpeg = QC * np.round(y / QC)
            X_compressed[i:i+8, j:j+8, 2] = idctn(y_jpeg)

    return ycbcr_to_rgb(X_compressed)



""" Cerinta 3 """
def mse_compression(X, mse_threshold=100, steps=30, color=True):

    min_, max_ = 1.0, 500.0
    image, s_ = None, None
    X = X.astype(np.float32)

    for _ in range(steps):

        s = (min_ + max_) / 2.0

        if color:
            X_compressed = compress_color_image(X, s)
        else:
            X_compressed = compress_monochrome_image(X, s)

        mse = np.mean((X - X_compressed.astype(np.float32)) ** 2)

        if mse <= mse_threshold:
            image = X_compressed
            s_ = s
            min_ = s      
        else:
            max_ = s       

    return s_, image



def cerinta1():
    X = datasets.ascent()
    X_compressed = compress_monochrome_image(X)

    plt.subplot(1,2,1)
    plt.imshow(X, cmap='gray')
    plt.title("original")
    plt.subplot(1,2,2)
    plt.imshow(X_compressed, cmap='gray')
    plt.title("comprimat")

    plt.savefig("Images/cerinta1.pdf", format="pdf")
    plt.show()



def cerinta2():
    X = datasets.face()
    X_compressed = compress_color_image(X, s=1)

    plt.subplot(1,2,1)
    plt.imshow(X)
    plt.title("original")
    plt.subplot(1,2,2)
    plt.imshow(X_compressed.astype(np.uint8))
    plt.title("comprimat")

    plt.savefig("Images/cerinta2.pdf", format="pdf")
    plt.show()



def cerinta3(mse):
    X = datasets.face()  
    s, X_compressed = mse_compression(X, mse)

    plt.imshow(X_compressed.astype(np.uint8))
    plt.title(f"mse = {mse} & s = {s:.2f}")
    plt.savefig(f"Images/cerinta3_{mse}.pdf", format="pdf")
    plt.show()




if __name__ == "__main__":
    #cerinta1()
    #cerinta2()
    cerinta3(250)