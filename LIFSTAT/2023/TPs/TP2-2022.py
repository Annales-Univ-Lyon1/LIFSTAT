import pandas as pan #pour les donnÃ©es :gestion des data.frame
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

#pip install pydataset
#from pydataset import data
#pydata=data()

###################Exercice 1#########################
#######1
def r(x,y):
  z = np.sqrt(x**2+y**2)
  return(z)
r(3,4)
##on a défini la norme euclidienne (longueur),
#puis on a calculé celle du vecteur (3,4)


######2
def g(N,p,k):
    z=np.math.factorial(N)/(np.math.factorial(k)*np.math.factorial(N-k))*(p**k)*((1-p)**(N-k))
    return(z)

g(20,0.5,10)
st.binom.pmf(10,n=20,p=0.5)#[1] 0.1761971
x=range(21)
plt.bar(x,height=[g(20,0.5,k) for k in x],width=0.1)
plt.plot(x,st.binom.pmf(x,n=20,p=0.5),'.',color="red")

###################Exercice 2#########################
########1
from datetime import datetime
N = 10**4
t0 = datetime.now().microsecond
s = 0
for i in range(1,N):
  if (i%2==0):
    s = s+i**2
#Pour tous les entiers pairs (congrus à 0 modulo 2),
#on ajoute leur carré, donc on fait la somme des
#carrés des entiers pairs entre 1 et N
t1 = datetime.now().microsecond
t = np.arange(1,N)
s2 = np.sum(t[t%2==0]**2)
t2 = datetime.now().microsecond
print("Valeurs obtenues :\n s = ",s," temps : ",t1-t0,"\n s2 = ",s2,"temps : ",t2-t1
    ,"\n" )
#Réponse chez moi:
#Valeurs obtenues :
#  s =  166716670000  temps :   10595 microsecondes
#s2 =  166716670000 temps :  1186 microsecondes

#On a comparé la vitesse de calcul de la somme 
#des carrés d'entiers pairs entre 1 et 10000, 
#soit par une boucle, soit en utilisant les 
#opérations sur les vecteurs de numpy, et la deuxième 
#est 10 fois plus rapide...


########2 et 3
N = 2*10**5#2k+1,k dans 0:99 ou  2k-1, k dans 1:100
t0 = datetime.now()
s = 0
for i in range(1,N):
  if (i%2==1):
    s = s+1/i**2
t1 = datetime.now()
t = np.arange(100000)
s2 = np.sum(1/(2*t+1)**2)
t2 = datetime.now()
print("Valeurs obtenues :\n s = ",s," temps : ",t1-t0,"\n s2 = ",s2,"temps : ",t2-t1
    ,"\n" )
#Valeurs obtenues :
# s =  1.2336980501361898  temps :  0:00:00.045075 
# s2 =  1.2336980501361696 temps :  0:00:00.000537 

#La deuxième méthode est 100 fois plus rapide !
#3 Bilan: il vaut mieux éviter les boucles en python!!


###################Exercice 3#########################
from scipy.stats import bernoulli, binom, geom, poisson;import scipy.stats as st

n, p = 10, 0.25
ProbasB=binom.pmf(range(n+1),n,p)# probabilites de la distribution binomiale
plt.bar(range(n+1),ProbasB,width=0.05)
plt.show()

#######1
n, p = 10, 0.5
plt.bar(range(n+1),binom.pmf(range(n+1),n,p),width=0.05)
plt.title("Diagramme en bâton de la loi B(10,0.5)")


n, p = 100, 0.25
plt.bar(range(51),binom.pmf(range(51),n,p),width=0.2)
plt.title("Diagramme en bâton de la loi B(100,0.25)")

n=10
plt.bar(range(n+1),poisson.pmf(range(n+1),2),width=0.05)
plt.title("Diagramme en bâton de la loi P(2)")
plt.show()


n=20
plt.bar(range(n+1),poisson.pmf(range(n+1),10),width=0.2)
plt.title("Diagramme en bâton de la loi P(10)")



n=5
plt.ylim((0,1))
plt.bar(range(n),geom.pmf(range(n),0.75),width=0.05)
plt.title("Diagramme en bâton de la loi G(0.75)")


n=20
plt.bar(range(n),geom.pmf(range(n),0.25),width=0.1)
plt.title("Diagramme en bâton de la loi G(0.25)")


n=6
plt.bar(range(1,n+1),np.repeat(1/6,6),width=0.05)
plt.title("Diagramme en bâton de la loi U({1,2,3,4,5,6})")

#partie simulation

n, p = 10, 0.5
b=binom.rvs(n,p,size=1000)
tab=pan.crosstab(b,columns="freq",normalize=True).values
x=range(int(min(tab)),int(min(tab))+len(tab))
plt.bar(x,np.reshape(tab,len(tab)),width=0.05)
plt.plot(x,binom.pmf(x,n,p),'r+')
plt.title("Diagramme en bâton d'une simulation de la loi B(10,0.5)")
#Autre solution
n, p = 10, 0.5
b=binom.rvs(n,p,size=1000)
tab=pan.DataFrame(b).value_counts(normalize=True,sort=False)
x=range(1,len(tab)+1)
plt.bar(x,tab,width=0.05)
plt.plot(x,binom.pmf(x,n,p),'r+')
plt.title("Diagramme en bâton d'une simulation de la loi B(10,0.5)")


mu= 10
p=poisson.rvs(mu,size=100000)
tab=pan.crosstab(p,columns="freq",normalize=True).values
x=range(int(min(tab)),int(min(tab))+len(tab))
plt.bar(x,np.reshape(tab,len(tab)),width=0.1)
plt.plot(x,poisson.pmf(x,mu),'r+')
plt.title("Diagramme en bâton d'une simulation de la loi P(10)")




#######2

n=25
plt.bar(np.arange(n+1)+.3,binom.pmf(np.arange(n+1),50,0.2),width=0.1,color="green")
plt.bar(np.arange(n+1)+.15,binom.pmf(np.arange(n+1),250,0.04),width=0.1,color="blue")
plt.bar(range(n+1),binom.pmf(range(n+1),500,0.02),width=0.1,color="red")
plt.plot(range(n+1),poisson.pmf(range(n+1),10),'o',color="black")
plt.title("Convergence vers la loi de Poisson P(10)")

###################Exercice 4#########################

#######1
n, p = 10, 0.25
plt.step(range(n+1),binom.cdf(range(n+1),n,p))
plt.title("Fonction de répartition de la loi B(10,0.25)")


#######2
n, p = 100, 0.5
plt.step(range(n+1),binom.cdf(range(n+1),n,p))
plt.title("Fonction cumulative de la loi B(100,0.5)")



n=15
plt.step(range(n+1),poisson.cdf(range(n+1),3))
plt.title("Fonction cumulative de la loi P(3)")

binom.cdf(3,50,0.2)#P(X\leq 3)=0.005656
1-binom.cdf(30,50,0.2)#P(X> 30)=1.1024325896613618e-10

###################Exercice 5#########################
#1
u = st.uniform.rvs(0,1,size=1000);u[1:10]
m = np.mean(u)
print("Moyenne empirique : ",m)

#On simule une vecteur de longueur 1000 noté u selon une variable uniforme sur [0,1]. On calcule ensuite sa moyenne empirique m puis on l'imprime dans une message.
#Elle est proche de la valeur théorique 0.5

#2
U = st.uniform.rvs(-3,6,size=10000);
V=U[U>0]

#3
plt.hist(V,density=True)
x=np.linspace(-1,4,500)
plt.plot(x,st.uniform.pdf(x,0,3),color="red")

#La loi de V est uniforme sur [0,3]

###################Exercice 6#########################
#1

U=st.uniform.rvs(size=10000,loc=-1,scale=2)
V =st.uniform.rvs(size=10000,loc=-1,scale=2)
S=U**2+V**2; Cas= (S<1)
U0= U[Cas]; V0 = V[Cas]; 

plt.axis('equal')#pour avoir les échelles identiques sur les coordonnées 

plt.plot(U,V,marker=',',linestyle='')
#la paire (U,V) est uniformément distribuée sur [-1,1]^2

#2


plt.axis('equal')
plt.plot(U0,V0,marker=',',linestyle='')
#la paire (U,V) est uniformément distribuée 
#sur le disque de centre 0 et de rayon 1

#3
#Le code simule U,V uniformément sur [-1,1] puis sélectionne les paires dans le disque unité 
#la longueur de U0 est de l'ordre du nombre de point dans le cercle, approximativement l'aire du disque divisé l'air 4 du carré fois le nombre de point total 10000
print(np.pi*10000/4)# de l'ordre de 7854
len(U0)


#4
R=np.sqrt(U0**2+V0**2)
plt.hist(R,density=True,bins=20)
x=np.linspace(0,1,100)
plt.plot(x,2*x,color="red")

#cela semble être la densité f(x)=2x1_{[0,1]}(x)


#5
X=np.sqrt(-np.log(R))*U0/R

np.mean(X)
np.std(X)
plt.hist(X,density=True)
x=np.linspace(-5,5,1000)
plt.plot(x,st.norm.pdf(x,loc=np.mean(X),scale=np.std(X)),color="red")


###################Exercice 7#########################
#1
def f(N,p):
  s = 0
  for i in range(N):
    if (st.uniform.rvs(0,1,1)<p):
      s = s+1
  return(s)

#On simule N nombres uniformément répartis entre 0
#et 1 (paramètres par défaut de runif), et on 
#compte combien sont en dessous de p. Cela revient 
#à simuler une variable bernoulli de paramètre p 
#et trouver la proportion

#2
N = 10**4
sum(bernoulli.rvs(size=N,p=0.2))
sum(binom.rvs(size=N,n=1,p=0.2))#B(n,p) size=1 donne Bernoulli
binom.rvs(n=N,size=1,p=0.2)#façon similaire en loi sans somme

#estimation du temps de calcul

N = 10**4
t0 = datetime.now()
s=f(N,0.2)
t1 = datetime.now()
s2 = sum(bernoulli.rvs(size=N,p=0.2))
t2 = datetime.now()
print("Valeurs obtenues :\n s = ",s," temps : ",t1-t0,"\n s2 = ",s2,"temps : ",t2-t1
    ,"\n" )

#Valeurs obtenues :
#    Valeurs obtenues :
#     s =  2004  temps :  0:00:00.679397 
#     s2 =  2026 temps :  0:00:00.004327 
#La fonction de scipy est au moins 100 fois plus rapide !
