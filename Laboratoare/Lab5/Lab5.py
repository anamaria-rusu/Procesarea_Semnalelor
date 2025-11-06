import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime



def exercitiul_1():

    # citire date
    x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=str)
   

    # PUNCTUL A
    # stocam datele (DateTime) in format datetime --> variabila date
    # calc dif dintre toate intervalele de timp consecutive 
    # 0.0002777777777777778
    date = [datetime.strptime(d, '%d-%m-%Y %H:%M') for d in x[:, 1]]
    diff_int = []
    for i in range (1,len(date)):
        # diferanta va fi in secunde 
        diff = (date[i] - date[i-1]).total_seconds()
        diff_int.append(diff)
    mean = np.mean(diff_int)
    fs = 1/mean
    print("punctul a: fs = ", fs)





    # PUNCTUL B
    # calc dif dintre ultima zi si prima zi de "masuratori"
    # aprox. 761 days 
    zi_0 = date[0]
    zi_n = date[-1]
    durata = (zi_n - zi_0)
    print("punctul b: durata = ", durata)





    # PUNCTUL C
    # nu avem aliasing <=> (Nyquist) fs > 2*f_max <=> f_max < fs/2
    # 0.0001388888888888889
    f_max = fs/2
    print("punctul c: f_max < ", f_max)





    # PUNCTUL D 
    # count_d - varibila ce retine numarul de masini 
    count_d = x[:, 2].astype(float)

    # scadem media din fiecare element
    # count_d = count_d - np.mean(count_d)
    N = len(count_d)

    # calculam FFT
    Xd = np.fft.fft(count_d)
    Xd = abs(Xd/N)
    Xd = Xd[:N//2]
    fd = fs * np.linspace(0, N//2, N//2) / N

    # afisarea pe scala logaritmica
    plt.yscale('log')
    plt.plot(fd, Xd)
    plt.savefig('Images/ex_d.pdf', format='pdf')
    plt.show()





    # PUNCTUL E
    # count_e - varibila ce retine numarul de masini
    count_e = x[:, 2].astype(float)

    # verificam media
    mean_e = np.mean(count_e)

    # comp cont <=> medie(semnal) <> 0
    # luam si o marja de 10^-10 
    # scadem media din fiecare element daca avem o comp cont
    if abs(mean_e) > 1e-10:
        count_e = count_e - mean_e
    N = len(count_e)

    # calculam FFT
    Xe = np.fft.fft(count_e)
    Xe = abs(Xe/N)
    Xe = Xe[:N//2]
    fe = fs*np.linspace(0, N//2, N//2) / N

    # afisarea pe scala logaritmica
    plt.yscale('log')
    plt.plot(fe, Xe)
    plt.savefig('Images/ex_e.pdf', format='pdf')
    plt.show()





    # PUNCTUL F
    # lucram cu Xe, fe de la punctul e
    ind =  np.argsort(Xe)[-4:][::-1] + 1
    print(ind)
    for i in ind:
        f0 = fe[i] 
        if f0 == 0:
            continue
        T_sec = 1 / f0
        T_h = T_sec / 3600
        T_day = T_sec / 86400
        T_week = T_sec / (7 * 86400)
        T_month = T_sec / (30 * 86400)
        print(f"f =  {f0:.3e} -> " f"T ≈ {T_sec:.2f}s = {T_h:.2f}h = {T_day:.2f} zile = {T_week:.3f} sapt = {T_month:.3f} luni")





    # PUNCTUL G
    # cautam o luna care incepe cu o zi de luni 
    found = False
    for i, t in enumerate(date):
        if i > 1000 and t.day == 1 and t.weekday() == 0:
            start = i
            found = True
            break
    if not found:
        start = 0

    # data de start a lunii 
    start_date = date[start]
    an, luna = start_date.year, start_date.month

    # luam toate datele din luna respectiva
    trafic_lunar = [count_e[j] for j, d in enumerate(date) if d.year == an and d.month == luna]
    plt.plot(trafic_lunar)
    plt.savefig('Images/ex_g.pdf', format='pdf')
    plt.show()





    # PUNCTUL H
    
    """
    
    - Presupun ca, la fel ca in Train.cvs, s-au facut masuratori in fiecare zi, la fiecare ora
    - Mai pp ca nu se stie nimic despre nicio data (adica nu avem vreo informatie de genul : pe 1 aug 2014 s-a facut o masuratoare)

    - Pentru inceut, as incerca sa determin ziua saptamanii in care a ineceput masuratoarea
    - Stim ca in timpul saptamnii (L-V) traficul este in general mai mare comparativ cu weekend-ul (S-D) 
    - Ar trebui sa cautam cea mai mare "discontinuitate" intre doua zile consecutive,
      pentru ca diferenta cea mai mare va fi de la duminica -> luni
    - Astfel determinam un tipar: 5 zile ridicate + 2 zile scazute + se repeta
    - Apoi, as lua (cat imi permite setul de date) intervale de cate 7 zile dupa pattern-ul de mai sus
    - Calculez media pentru fiecare zi a saptamnii 
    - Compar ziua inceperii masuratorii cu aceste medii, si o aleg pe cea mai apropiata
    - Astfel, am identificat ziua saptamanii

    - Solutia are limitari pentru ca putem determina doar ziua saptamnaii, ci nu data exacta
    - Acuratetea depinde de anomaliile care pot aparea : schimbarea orei, sarbatorile, zilele libere etc

    
    ????  Cateva extensii care pot imbuntatii solutia (desi putin probabaile): ????
    - daca am avea mai multe masuratori, sa zicem pe cel putin 6 luni (si stim tara in care au fost facute masuratorile) 
      ne putem folosi de schimbarea de ora
    - daca identifica o zi cu 23h / 25h putem gasi un punct de referinta mai concret si sa stabilim eventual si luna 
    - fiind un ev rar, am obtine destule info
    - dar, solutia ar fi valabila doar daca se garanteaza ca masuratorile sunt la intervale de timp egale 

    """





    # PUNCTUL I
    # pentru o perioada de 7 zile ( 7 zile * 24 ore * 3600 s) -> tendinta saptamnala 
    ff = 1 / (7 * 24 * 3600)  
    plt.plot(fd[fd < ff], Xd[fd < ff])
    plt.savefig('Images/ex_i.pdf', format='pdf')
    plt.show()





if __name__ == "__main__":
    exercitiul_1()

