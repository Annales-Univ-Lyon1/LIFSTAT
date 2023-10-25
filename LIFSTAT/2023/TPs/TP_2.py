#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:53:55 2023

@author: virgilebroillet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats.mstats as ms
from pydataset import data


data('cars',show_doc=True)
cars=data('cars')

"""
print("la moyenne de la vitesse des voitures est de : ", (np.mean(cars.speed)))
print("la moyenne de la distance des voitures est de : ", (np.mean(cars.dist)))
print("la variance empirique de la vitesse des voitures est de : ", np.var(cars.speed))
print("la variance empirique de la distance des voitures est de : ", np.var(cars.dist))
print("la variance empirique non biaisée de la vitesse :",np.var(cars.speed, ddof=1)) #variance empirique nb de speed
print("la variance empirique non biaisée de la distance :",np.var(cars.dist, ddof=1)) #variance empirique nb de speed
print("la médiane de la vitesse :",np.median(cars.speed)) 
print("la médiane de la distance :",np.median(cars.dist))
print("l'écart type de la vitesse :",np.std(cars.speed)) 
print("l'écart type de la distance :",np.std(cars.dist))

for i in [0.25, 0.5, 0.75]: # j'effectue une division par quatre pour avoir les quartiles aux positions 0.25, 0.5, 0.75
    print("l'écart type : ", i, " de la vitesse ", np.quantile(cars.speed, i, method="lower"))
    
for i in [0.25, 0.5, 0.75]:
    print("l'écart type : ", i, " de la distance ", np.quantile(cars.dist, i, method="lower"))
"""

print(cars.describe());

plt.hist(cars.speed, color="blue")
plt.title("Histogramme de la vitesse des voitures en fonction de leur nombres")
plt.xlabel("Vitesses des voitures")
plt.ylabel("Nombre de voiture a la vitesse x")
#plt.show()

data('iris',show_doc=True)
iris = data('iris')

plt.boxplot(iris["Sepal.Length"])
plt.show()

x = np.array([1,8,5,1])

y = [0]
for i in range(1, 10, 2):
    y += [i]
print(y)

#affiche la valuer d'indice 4 dans le tableau y
print(y[4])
#affiche toutes les valeurs dont l'indice est plus grand que 2 (inclus) et plus petit que 4 (exclus)
print(y[2:4])
#affiche l'élement en d'indice MAX - 2
print(y[-2])

y1 = []
for i in range(0, len(y), 2):
    y1 += [y[i]]
print(y1)

y2 = []
for i in range(0, len(y)):
    if y[i] > 0:
        y2 += [y[i]]
print(y2)


X=np.repeat(x.reshape(1,4),25,axis=0).reshape(100)
print(X)
    



