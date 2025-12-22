import numpy as np
import matplotlib.pyplot as plt
import cv2 
from scipy.fft import dctn, idctn
from scipy import datasets



"""
- clasa JpegCompressor implementeaza algoritmul de comprimare JPEG pentru imagini monocrome si color


- ca si variabile, clasa contine:
    - Q_jpeg        =  var de clasa, matricea de cuantizare standard JPEG 
    - X             = var de instanta, imaginea initiala 
    - X_compressed  = var de instanta, imaginea comprimata

    
- ca si metode, clasa contine:
    - rgb_to_ycbcr               = conversia unei imagini RGB in YCbCr 
    - ycbcr_to_rgb               = conversia unei imagini YCbCr in RGB
    - compress_monochrome_image  = comprimarea unei imagini monocrome
    - compress_color_image       = comprimarea unei imagini color 
    - mse_compression            = comprimarea unei imagini pana la un MSE dat 
    - video_compression          = comprimarea unui fisier video 

- conversiile RGB <-> YCbCr confoma ITU-R BT.601 : https://web.archive.org/web/20180421030430/http://www.equasys.de/colorconversion.html

"""


class JpegCompressor:

    
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



    # initializare : X = iamgea , X_compressed = imaginea comprimata
    def __init__(self, image):
        self.X = image
        self.X_compressed = np.zeros_like(image)

    

    # conversie RGB -> YCbCr 
    def __rgb_to_ycbcr(self):

        X_ycbcr = np.zeros_like(self.X, dtype=np.float32)
        h, w, _ = self.X.shape

        for i in range(h):
            for j in range(w):

                R = float(self.X[i, j, 0])
                G = float(self.X[i, j, 1])
                B = float(self.X[i, j, 2])

                Y  = 16  + 0.257 * R + 0.504 * G + 0.098 * B
                Cb = 128 - 0.148 * R - 0.291 * G + 0.439 * B
                Cr = 128 + 0.439 * R - 0.368 * G - 0.071 * B

                X_ycbcr[i, j, 0] = np.clip(Y,  16, 235)
                X_ycbcr[i, j, 1] = np.clip(Cb, 16, 240)
                X_ycbcr[i, j, 2] = np.clip(Cr, 16, 240)

        return X_ycbcr
    


    # conversie YCbCr -> RGB 
    def __ycbcr_to_rgb(self):

        X_rgb = np.zeros_like(self.X_compressed, dtype=np.float32)
        h, w, _ = self.X_compressed.shape

        for i in range(h):
            for j in range(w):

                Y  = float(self.X_compressed[i, j, 0]) - 16.0
                Cb = float(self.X_compressed[i, j, 1]) - 128.0
                Cr = float(self.X_compressed[i, j, 2]) - 128.0

                R = 1.164 * Y + 1.596 * Cr
                G = 1.164 * Y - 0.392 * Cb - 0.813 * Cr
                B = 1.164 * Y + 2.017 * Cb

                X_rgb[i, j, 0] = np.clip(R, 0, 255)
                X_rgb[i, j, 1] = np.clip(G, 0, 255)
                X_rgb[i, j, 2] = np.clip(B, 0, 255)

        return X_rgb



    """ Cerinta 1 """
    # comprimarea unei imagini monocrome
    def compress_monochrome_image(self, Q = None):
        
        # daca nu e specificata o matrice de cuantizare, se foloseste cea standard
        if Q is None:
            Q = self.Q_jpeg

        h, w = self.X.shape

       # daca h, w nu sunt multipli de 8, atunci "restul" de pixeli, ramane 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):

                # se ia fiecare pachet de 8x8
                x = self.X[i:i+8, j:j+8]
                y = dctn(x)
                y_jpeg = Q * np.round(y / Q)
                x_jpeg = idctn(y_jpeg)

                # se insereaza inapoi in imaginea comprimata
                self.X_compressed[i:i+8, j:j+8] = x_jpeg



    # comprimarea unei imagini color
    def compress_color_image(self, Q = None):

        # daca nu e specificata o matrice de cuantizare , se foloseste cea standard
        if Q is None:
            Q = self.Q_jpeg

        # se converteste imaginea in YCbCr
        X_ycbcr = self.__rgb_to_ycbcr()
        h, w , _ = self.X.shape

        for c in range(3):  
            for i in range(0, h, 8):
                for j in range(0, w, 8):

                    # se ia fiecare pachet de 8x8 pentru fiecare canal
                    x = X_ycbcr[i:i+8, j:j+8, c]    
                    y = dctn(x)
                    y_jpeg = Q * np.round(y / Q)
                    x_jpeg = idctn(y_jpeg)
                    # se insereaza inapoi in imaginea comprimata
                    self.X_compressed [i:i+8, j:j+8, c] = x_jpeg

        # se converteste inapoi in RGB
        self.X_compressed = self.__ycbcr_to_rgb()




    # comprimarea unei imagini pana la un MSE dat ca prag
    def mse_compression(self, mse_threshold=1500, color=True, steps=20):
       
        # cautam scalarea optima folosind cautarea binara
        low, high = 0.1, 1500
        best_img = None
        best_s = None

        for _ in range(steps):
           
            s = (low + high) / 2.0
            Qs = s * self.Q_jpeg

            # se ia pe cazuri, in functie de tipul imaginii
            if not color:
                self.compress_monochrome_image(Qs)
            else:
                self.compress_color_image(Qs)

            # calculam MSE
            mse = np.mean((self.X.astype(np.float32) - self.X_compressed.astype(np.float32)) ** 2)

            if mse <= mse_threshold:
                best_img = self.X_compressed.copy()
                best_s = s
                low = s          
            else:
                high = s         

        return best_s, best_img




    def video_compression(self, input_path, output_path):
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # rezolutie mica (rapid)
        target_w, target_h = 960, 540
        target_h = (target_h // 8) * 8   # 536
        target_w = (target_w // 8) * 8   # 960

        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

        Qs = 40 * self.Q_jpeg  # mare = blur

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_bgr = cv2.resize(frame_bgr, (target_w, target_h))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            comp = JpegCompressor(frame_rgb)
            comp.compress_color_image(Qs)

            frame_out = np.clip(comp.X_compressed, 0, 255).astype(np.uint8)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
            out.write(frame_out)

        cap.release()
        out.release()


    # compressor = JpegCompressor(datasets.face())
    # compressor.video_compression(
    #     input_path="D:\\GitHubRepo\\Procesarea_Semnalelor\\Tema\\raton.mp4",
    #     output_path="D:\\GitHubRepo\\Procesarea_Semnalelor\\Tema\\raton_compressed.mp4"
    # )     
    #     plt.savefig(f"Images/ex3.pdf", format='pdf')
  
            


def cerinata_1():
    # imagea din cerinta 
    X = datasets.ascent().astype(np.float32)
    mono = JpegCompressor(X)
    mono.compress_monochrome_image()

if __name__ == "__main__":

    # 
    

    monochrome_compression = JpegCompressor(X)
    monochrome_compression.compress_monochrome_image(Q=10 * monochrome_compression.Q_jpeg)
    Y = monochrome_compression.X_compressed

    # coordonate patch-uri 8x8
    patch_coords = [
        (0, 0),
        (64, 64),
        (128, 128)
    ]

    # extragere patch-uri
    patches_X = [X[i:i+8, j:j+8] for (i, j) in patch_coords]
    patches_Y = [Y[i:i+8, j:j+8] for (i, j) in patch_coords]

    # salvare patch-uri
    plt.imsave("patch_X_1.png", patches_X[0], cmap="gray", vmin=0, vmax=255)
    plt.imsave("patch_Y_1.png", patches_Y[0], cmap="gray", vmin=0, vmax=255)

    plt.imsave("patch_X_2.png", patches_X[1], cmap="gray", vmin=0, vmax=255)
    plt.imsave("patch_Y_2.png", patches_Y[1], cmap="gray", vmin=0, vmax=255)

    plt.imsave("patch_X_3.png", patches_X[2], cmap="gray", vmin=0, vmax=255)
    plt.imsave("patch_Y_3.png", patches_Y[2], cmap="gray", vmin=0, vmax=255)




    



        
        
