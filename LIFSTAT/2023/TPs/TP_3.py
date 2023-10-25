#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:57:16 2023

@author: virgilebroillet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Question 1
def r(x,y):
    z = np.sqrt(x**2+y**2)
    print(z)
r(3,4)

def Bino(n, p, k):
    bino = ((np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n-k))) * ((p ** k) * ((1-p) ** (n - k))))
    print(bino)
    
print(stats.binom.pmf(3, 2, 1))

def inverse(n):
    s = 0
    for i in range(1,n):
        if (i%2!=0):
            s += 1/(i**2)
    print(s)

inverse(100000)

from scipy.stats import bernoulli, binom, geom, poisson;
import scipy.stats as st

n, p = 10, 0.25;
ProbasB=binom.pmf(range(n+1),n,p) #probabilites de la distribution binomiale
plt.bar(range(n+1),ProbasB,width=0.05)
plt.title("Diagramme en bâton de la loi B(10,0.25)")
plt.show();

n, p = 10, 0.5;
ProbasB=binom.pmf(range(n+1),n,p) #probabilites de la distribution binomiale
plt.bar(range(n+1),ProbasB,width=0.05)
plt.title("Diagramme en bâton de la loi B(10,0.5)")
plt.show();

n, p = 100, 0.25;
ProbasB=binom.pmf(range(n+1),n,p) #probabilites de la distribution binomiale
plt.bar(range(n+1),ProbasB,width=0.05)
plt.title("Diagramme en bâton de la loi B(100,0.25)")
plt.show();

n, p = 10, 2
ProbasP=poisson.pmf(range(n+1),p) #probabilites de la distribution de la loi de poisson
plt.bar(range(n+1),ProbasP,width=0.05)
plt.title("Diagramme en bâton de la loi P(2)")
plt.show();

n, p = 10, 10
ProbasP=poisson.pmf(range(n+1),p) #probabilites de la distribution de la loi de poisson
plt.bar(range(n+1),ProbasP,width=0.05)
plt.title("Diagramme en bâton de la loi P(2)")
plt.show();

n, g = 5, 0.75
plt.ylim((0,1))
plt.bar(range(n),geom.pmf(range(n),g),width=0.05)
plt.title("Diagramme en bâton de la loi G(0.75)")
plt.show()


n, g = 20, 0.25
plt.bar(range(n),geom.pmf(range(n),g),width=0.1)
plt.title("Diagramme en bâton de la loi G(0.25)")
plt.show()


n=6
plt.bar(range(1,n+1),np.repeat(1/6,n),width=0.05)
plt.title("Diagramme en bâton de la loi U({1,2,3,4,5,6})")
plt.show()

n, p = 10, 0.5
b=binom.rvs(n,p,size=1000)
tab=pd.crosstab(b,columns="freq",normalize=True).values
x=range(int(min(tab)),int(min(tab))+len(tab))
plt.bar(x,np.reshape(tab,len(tab)),width=0.05)
plt.plot(x,binom.pmf(x,n,p),'r+')
plt.title("Diagramme en bâton d'une simulation de la loi B(10,0.5)")
plt.show()

mu= 10
p=poisson.rvs(mu,size=100000)
tab=pd.crosstab(p,columns="freq",normalize=True).values
x=range(int(min(tab)),int(min(tab))+len(tab))
plt.bar(x,np.reshape(tab,len(tab)),width=0.1)
plt.plot(x,poisson.pmf(x,mu),'r+')
plt.title("Diagramme en bâton d'une simulation de la loi P(10)")
plt.show()




#######2

n=25
plt.bar(np.arange(n+1)+.3,binom.pmf(np.arange(n+1),50,0.2),width=0.1,color="green")
plt.bar(np.arange(n+1)+.15,binom.pmf(np.arange(n+1),250,0.04),width=0.1,color="blue")
plt.bar(range(n+1),binom.pmf(range(n+1),500,0.02),width=0.1,color="red")
plt.plot(range(n+1),poisson.pmf(range(n+1),10),'o',color="black")
plt.title("Convergence vers la loi de Poisson P(10)")
plt.show()

###################Exercice 4#########################

#######1
n, p = 10, 0.25
plt.step(range(n+1),binom.cdf(range(n+1),n,p))
plt.title("Fonction de répartition de la loi B(10,0.25)")
plt.show()


#######2
n, p = 100, 0.5
plt.step(range(n+1),binom.cdf(range(n+1),n,p))
plt.title("Fonction cumulative de la loi B(100,0.5)")
plt.show()

#######2
n, p = 1000, 0.5
plt.step(range(n+1),binom.cdf(range(n+1),n,p))
plt.title("Fonction cumulative de la loi B(1000,0.5)")
plt.show()

n=15
plt.step(range(n+1),poisson.cdf(range(n+1),3))
plt.title("Fonction cumulative de la loi P(3)")
plt.show()

binom.cdf(3,50,0.2)#P(X\leq 3)=0.005656
1-binom.cdf(30,50,0.2)#P(X> 30)=1.1024325896613618e-10
