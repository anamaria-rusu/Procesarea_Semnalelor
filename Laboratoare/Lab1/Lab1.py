import numpy as np
import matplotlib.pyplot as plt

def exercitiul1(start, end, pas):
   # semnalele continue (din ex1) --> s(t) = cos(2 * pi * t + phi)
   # start = de la ce valoare sa inceapa t
   # end = pana la ce valoare sa ajunga t
   # pas = pasul de esantionare

   # subpunct a
   # axa reala de timp (continuu) (expl : se creeaza un vector t cu "puncte" de la 0 la 0.03 (sec) cu progresia de 0.0005)
   t = np.linspace(start, end, int((end - start) / pas))


   # subpunct b
   # semnalele continue 
   xt = np.cos(520 * np.pi * t + np.pi / 3)
   yt = np.cos(280 * np.pi * t - np.pi / 3)
   zt = np.cos(120 * np.pi * t + np.pi / 3)

   # afisare semnale continue in subplot-uri
   semnale = [xt, yt, zt]
   titluri = ["x(t)","y(t) ","z(t)"]
   fig, axs = plt.subplots(len(semnale), 1, figsize=(10, 8))
   for i in range(len(semnale)):
      axs[i].plot(t, semnale[i])
      axs[i].set_title(titluri[i])
      axs[i].grid()
   plt.tight_layout(rect=[0, 0, 1, 1])
   plt.savefig("Images/ex1_ab.pdf", format='pdf')
   plt.show()


   # subpunct c
   # pentru intervalul 0, 0.3 --> avem 0.3 secunde | fs = 200 Hz | numarPuncte = 0.3 * 200 = 60
   fs = 200
   t = np.linspace(start, end, int((end - start)* fs))
   xn = np.cos(520 * np.pi * t + np.pi / 3)
   yn = np.cos(280 * np.pi * t - np.pi / 3)
   zn = np.cos(120 * np.pi * t + np.pi / 3)
   semnale = [xn, yn, zn]

   fig, axs = plt.subplots(len(semnale),1, figsize=(10, 8))
   for i in range(len(semnale)):
      axs[i].stem(t,semnale[i])
      axs[i].set_title(titluri[i])
   plt.tight_layout(rect=[0, 0, 1, 1])
   plt.savefig("Images/ex1_c.pdf", format='pdf')
   plt.show()





def exercitiul2():

   def plot_discret(t, x, title):
      plt.stem(t, x)
      plt.grid()
      plt.savefig("Images/ex2_"+ title +".pdf", format='pdf')
      plt.show()
   
   def plot_continuu(t,x,title):
      plt.plot(t, x)
      plt.grid()
      plt.savefig("Images/ex2_"+ title +".pdf", format='pdf')
      plt.show()


   # subpunct a
   # sinusoidal | f = 400 Hz | #e = 1600 
   # f * durata = #e --> durata = 1600 / 400 = 4 sec
   t = np.linspace(0,4,1600)
   a = np.sin(2 * np.pi * 400 * t)
   plot_discret(t, a ,"a")


   # subpunct b
   # sinusoidal | f = 800 Hz | durata = 3 sec
   t = np.linspace(0,3, 3*800)
   b = np.sin(2 * np.pi * 800 * t)
   plot_continuu(t, b ,"b")


   # subpunct c
   # sawtooth | f = 240 Hz 
   # sawtooth -- > semnal periodi - linie care urca constant si apoi scade brusc 
   duration = 6 / 240 
   t = np.linspace(0, duration, 25)
   c = 240 * t - np.floor(240 * t) 
   plot_continuu(t, c ,"c")


   # subpunct d
   # square | f = 300 Hz
   t = np.linspace(0,5, 5*300)
   d = np.sign(np.sin(2 * np.pi * 300 * t))
   plot_continuu(t, d ,"d")


   # subpunct e
   # 2D random 2^7 x 2^7
   e = np.random.rand(2**7, 2**7)
   plt.imshow(e)
   plt.grid()
   plt.savefig("Images/ex2_e.pdf", format='pdf')
   plt.show()


   # subpunct f
   # 2D 2^7 x 2^7
   # semnal --> initializat cu pixelii dintr-o imagine 128 x 128 
   f = plt.imread("img.jpg")
   f = f[:, :, 0]  
   f = f / 255.0   
   plt.imshow(f, cmap='gray')
   plt.grid()
   plt.savefig("Images/ex2_f.pdf", format='pdf')
   plt.show()




def exercitiul3():
   """
   semnal --> digitalizat
   fs = 2000 Hz

   a.interval de timp intre doua esantioane
   T = 1 / fs = 1 / 2000 = 0.0005 s

   b.esantion --> 4 biti
   fs = 2000 Hz <=> 2000 esantioane / secunda
   1 h = 3600 secunde
   deci 2000*3600*4 = 28800000 b / ora = 3.6 Mb 

   """
   return



if __name__ == "__main__":
   #exercitiul1(0, 0.3, 0.0005)
   exercitiul2()
   #pentru exercitiul3 rezolvarea e in functia exercitiul3() in comentarii
   pass





   
