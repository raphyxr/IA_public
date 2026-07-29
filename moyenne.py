chiffres = [4.8345,
5.87675,
7.79486,
5.7094,
5.45465,
6.33266,
5.82711,
5.02887,
7.08469,
4.4266,
6.62119,
7.027]

chiffres_2 = [7.2720,
6.6867,
5.74907,
5.0890,
5.2866,
6.87144,
7.1828,
7.3134,
5.3859,
8.0323,
5.2601,
4.35250]

chiffres_3 = [4.8652,
4.6207,
7.91338,
6.1087,
5.2545,
6.02757,
4.89604,
5.5939]

def moyenne (liste):
    print ("calcul moyenne")
    div = len(liste)
    dive = 0
    global nb
    for i in range(len(liste)):
        dive = int(dive) + int(i)
    moyenne = dive / div
    print ("la moyenne numéro " + str(nb) + " est : ", moyenne)


if __name__ == "__main__":
    nb = 1
    moyenne(chiffres)
    nb = 2
    moyenne(chiffres_2)
    nb = 3
    moyenne(chiffres_3)
