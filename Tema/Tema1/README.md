### Procesarea Semnalelor - Tema 1 (Compresie JPEG)

Acest proiect conține rezolvările pentru **Tema 1 - compresia JPEG**.

## Structura directorului

* `tema1.ipynb` – conține cerințele temei.
* `jpeg.py` – implementarea soluțiilor.
* `Images/` – conține fișierele cu rezultatele obținute (PDF pentru imagini, MP4 pentru video) și video-ul original pentru cerința 4.

## Descrierea fișierului `jpeg.py`

* **Matrici de cuantizare**

  * `Q_jpeg` – pentru imagini monocrome / componenta de luminanță.
  * `Q_jpeg_chrominance` – pentru imagini color / componentele de crominanță.
    Valorile sunt preluate conform **ITU-T Recommendation T.81 / ISO/IEC 10918-1** (pagina 147).

* **Funcții auxiliare**

  * `rgb_to_ycbcr` și `ycbcr_to_rgb` – pentru conversia între RGB și YCbCr.

* **Funcții principale**

  1. `compress_monochrome_image` – comprimă o imagine monocromă.
  2. `compress_color_image` – comprimă o imagine color.
  3. `mse_compression` – comprimă o imagine până la un MSE dorit.
  4. Funcții pentru cerințele 1-4:

     * `cerinta1()` – testează compresia imaginilor monocrome (testat pe `datasets.ascent()`).
     * `cerinta2()` – testează compresia imaginilor color (testat pe `datasets.face()`).
     * `cerinta3()` – testează compresia pentru diferite valori de MSE. Fișierele rezultate sunt salvate cu sufixul corespunzător MSE-ului (testat pe `datasets.face()`).
     * `cerinta4()` – comprimă un video.

* **Main** – apelează funcțiile `cerinta1()`–`cerinta4()` pentru generarea tuturor rezultatelor.

## Rezultate

* PDF-uri pentru cerințele 1-3 (imagini).
* MP4 pentru cerința 4 (video).

## Resurse

* [ITU-T Recommendation T.81 / ISO/IEC 10918-1 (JPEG Standard)](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) 
* [Color Conversion Reference](https://web.archive.org/web/20180421030430/http://www.equasys.de/colorconversion.html)
