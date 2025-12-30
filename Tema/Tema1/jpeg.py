import numpy as np
import matplotlib.pyplot as plt
import cv2 
from scipy.fft import dctn, idctn
from scipy import datasets

# matricea de cuantizare (pentru imaginile monocrome)
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

# matricea de cuantizare (pentru componentele cromatice)
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

    # rezultatul YCbCr in urma conversiei
    X_ycbcr = np.zeros_like(X, dtype=np.float32)
    h, w, _ = X.shape

    # se parcurge fiecare pixel 
    for i in range(h):
        for j in range(w):

            # se extrag componentele R G B
            R = float(X[i, j, 0])
            G = float(X[i, j, 1])
            B = float(X[i, j, 2])

            # se realizeaza conversia
            Y  = 16  + 0.257 * R + 0.504 * G + 0.098 * B
            Cb = 128 - 0.148 * R - 0.291 * G + 0.439 * B
            Cr = 128 + 0.439 * R - 0.368 * G - 0.071 * B

            # se stocheaza valorile in matricea YCbCr
            X_ycbcr[i, j, 0] = np.clip(Y,  16, 235)
            X_ycbcr[i, j, 1] = np.clip(Cb, 16, 240)
            X_ycbcr[i, j, 2] = np.clip(Cr, 16, 240)

    return X_ycbcr



# conversie YCbCr -> RGB 
def ycbcr_to_rgb(X_compressed):

    # rezultatul RGB in urma conversiei
    X_rgb = np.zeros_like(X_compressed, dtype=np.float32)
    h, w, _ = X_compressed.shape

    # se parcurge fiecare pixel
    for i in range(h):
        for j in range(w):

            # se extrag componentele Y Cb Cr
            # se realizeaza conversia
            Y  = float(X_compressed[i, j, 0]) - 16.0
            Cb = float(X_compressed[i, j, 1]) - 128.0
            Cr = float(X_compressed[i, j, 2]) - 128.0

            R = 1.164 * Y + 1.596 * Cr
            G = 1.164 * Y - 0.392 * Cb - 0.813 * Cr
            B = 1.164 * Y + 2.017 * Cb

            # se stocheaza valorile in matricea RGB
            X_rgb[i, j, 0] = np.clip(R, 0, 255)
            X_rgb[i, j, 1] = np.clip(G, 0, 255)
            X_rgb[i, j, 2] = np.clip(B, 0, 255)

    return X_rgb



""" Cerinta 1 - comprimarea iamginilor monocrome """
def compress_monochrome_image(X, s = 1):

    # s = factorul de scalare a matricii de cuantizare, by default egal cu 1 
    # Q = matricea de cuantizare scalata
    # se va utiliza doar matricea Q_jpeg pentru ca imaginea e monocroma (componenta de luminozitate)

    Q = s * Q_jpeg
    h, w = X.shape

    # se ajusteaza dimensiunile imaginii pentru a fi multipli de 8 (dimensiunea blocului, impartirea e exacta)
    h = (h // 8) * 8
    w = (w // 8) * 8
    X = X[:h, :w]  

    # imaginea comprimata
    X_compressed = np.zeros_like(X, dtype=np.float32)

    # se parcurge fiecare bloc de 8x8 si se realizeaza compresia    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            x = X[i:i+8, j:j+8]
            y = dctn(x)
            y_jpeg = Q * np.round(y / Q)
            x_jpeg = idctn(y_jpeg)
            X_compressed[i:i+8, j:j+8] = x_jpeg

    return X_compressed



""" Cerinta 2 - comprimarea iamginilor color """
def compress_color_image(X, s=1):
    
    # s = factorul de scalare a matricii de cuantizare, by default egal cu 1 
    # QY = matricea de cuantizare scalata pentru componenta de luminozitate (Y)
    # QC = matricea de cuantizare scalata pentru componentele cromatice (Cb si Cr)

    QY = s * Q_jpeg
    QC = s * Q_jpeg_chrominance
    h, w, _ = X.shape

    # se ajusteaza dimensiunile imaginii pentru a fi multipli de 8 (dimensiunea blocului, impartirea e exacta)
    h = (h // 8) * 8
    w = (w // 8) * 8
    X = X[:h, :w]

    # se realizeaza conversia RGB -> YCbCr
    X_ycbcr = rgb_to_ycbcr(X).astype(np.float32)
    # imaginea comprimata in spatiul YCbCr
    X_compressed = np.zeros_like(X_ycbcr, dtype=np.float32)

    # se parcurge fiecare bloc de 8x8 si se realizeaza compresia 
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

    # se realizeaza conversia YCbCr -> RGB si se ret rezultatul
    return ycbcr_to_rgb(X_compressed)



""" Cerinta 3 - comprimarea imaginilor cu MSE dat """
def mse_compression(X, mse_threshold=100, steps=30, color=True):

    # se cauta factorul de scalare optim s folosind cautarea binara
    # presupunam ca s se afla in intervalul [min_, max_]
    min_, max_ = 1.0, 500.0
    image, s_ = None, None
    X = X.astype(np.float32)

    # presupunem ca facem maxim steps iteratii
    for _ in range(steps):
        s = (min_ + max_) / 2.0

        # imaginea poate fi color sau monocroma
        if color:
            X_compressed = compress_color_image(X, s)
        else:
            X_compressed = compress_monochrome_image(X, s)

        # se calculeaza MSE intre imaginea originala si cea comprimata
        mse = np.mean((X - X_compressed.astype(np.float32)) ** 2)

        # se ajusteaza intervalul de cautare in functie de mse
        if mse <= mse_threshold:
            image = X_compressed
            s_ = s
            min_ = s      
        else:
            max_ = s       

    return s_, image



""" Cerinta 4 - comprimarea unui video """
def cerinta4(input_path, output_path, s=200, scale = 1/3):

    # se citeste inputul (un video de pe disc)
    input = cv2.VideoCapture(input_path)
    if not input.isOpened():
        raise ValueError(f"cannot open input video: {input_path}")

    fps = input.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-1:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # se scade rezolutia (pt cost computational mai mic...), se ajusteaza dimensiunile pentru a fi multipli de 8
    w = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(w * scale)
    h = int(h * scale)
    h = (h // 8) * 8
    w = (w // 8) * 8

    output = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not output.isOpened():
        raise ValueError(f"cannot create output video: {output_path}")

    # se parcurg toate frame-urile din video
    # pentru fiecare frame se realizeaza compresia 
    while True:
        ret, frame_bgr = input.read()
        if not ret:
            break

        frame_bgr = cv2.resize(frame_bgr, (w, h))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb_comp = compress_color_image(frame_rgb.astype(np.float32), s=s)
        frame_out = np.clip(frame_rgb_comp, 0, 255).astype(np.uint8)
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        output.write(frame_out)

    # se elibereaza resursele
    input.release()
    output.release()



""" Apelurile functiilor pentru cerintele 1,2,3 """

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



def cerinta3(mse):
    X = datasets.face()  
    s, X_compressed = mse_compression(X, mse)

    plt.imshow(X_compressed.astype(np.uint8))
    plt.title(f"mse = {mse} & s = {s:.2f}")
    plt.savefig(f"Images/cerinta3_{mse}.pdf", format="pdf")



# Main
if __name__ == "__main__":
    cerinta1()
    cerinta2()
    cerinta3(100)
    cerinta3(200)
    cerinta3(300)
    cerinta3(400)
    cerinta4("Images\\cerinta4_raton_input.mp4", "Images\\cerinta4_raton_output.mp4")