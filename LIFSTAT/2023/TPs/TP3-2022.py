import pandas as pan #pour les donnÃ©es :gestion des data.frame
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


#pip install pydataset
#from pydataset import data
#pydata=data()

###################Exercice 1#########################
#######3.1
##1
N=2000
X=st.binom.rvs(1,0.5,size=N)

##1
Xbar=np.cumsum(X)/(1+np.arange(N))
plt.plot(Xbar)
plt.plot(np.repeat(0.5,N),color="red")
plt.title("Illustration de la LGN pour B(1/2)")

#autre solution
Xbar=[np.mean(X[:(n+1)]) for n in range(N)]

##2
N=2000
X=st.norm.rvs(1,2,size=N)#Attention donner écart-type 2! et pas 4
Xbar=np.cumsum(X)/(1+np.arange(N))
plt.plot(Xbar)
plt.plot(np.repeat(1,N),color="red")
plt.title("Illustration de la  LGN pour N(1,4)")

##3 
N=10000
C=st.cauchy.rvs(loc=0,scale=1,size=N)
Cbar=np.cumsum(C)/(1+np.arange(N))
plt.plot(Cbar)
plt.plot(np.repeat(0,N),color="red")
plt.title("Illustration de l'absence de LGN pour C(0,1)")

#On remarque même avec un échantillon assez grand, il y a des sauts très loin de la médiane 0.
# C'est un cas où la loi des grands nombre ne s'applique pas, la variable
# n'a pas de moyenne vers laquelle la moyenne empirique pourrait tendre.
# Attention, il y a certaines simulations où cela se voit moins, répétez la simulation plusieurs fois.

###################Exercice 3.2#########################
##1
N=10000
M=5000
p=0.5
S=st.binom.rvs(N,p,size=M)
def varTCL(x):
    return((x/N-p)/np.sqrt(p*(1-p)/N))
V=[varTCL(x) for x in S]

## 2
plt.hist(V,bins=20,density=True)
x=np.linspace(start=-3,stop=3,num=600)
plt.plot(x,st.norm.pdf(x),color="red")


###################Exercice 3.3#########################



taille=[180, 170, 186, 184, 182, 171, 184, 167, 180, 177]

#1
#On doit supposer que la taille des étudiants suit 
#une loi normale 
#(ce qu'on ne peut pas vérifier avec un échantillon si petit)

#2
n = len(taille)
# Calcule de la moyenne et delta à 95%
moyenne = np.mean(taille)
sigma = np.sqrt(5)
delta_95 = (st.norm.ppf(0.975)*sigma) / np.sqrt(n)
# IC associé
IC_95 = (moyenne - delta_95, moyenne + delta_95)
print("IC au niveau 0.95 :[%s,%s]" % IC_95)
#IC - 95%:[ 176.714 , 179.486 ]

#z_{\alpha/2} du cours est st.norm.ppf(0.975)

# 3
taille = x=np.concatenate((taille,[173, 167, 170, 174, 178, 178, 175, 166, 172, 170]))
n = len(taille)
# Calcule de la moyenne et delta à 95%
moyenne = np.mean(taille)
sigma = np.sqrt(5)
delta_95 = (st.norm.ppf(0.975)*sigma) / np.sqrt(n)
# IC associé
IC_95 = (moyenne - delta_95, moyenne + delta_95)
print("IC au niveau 0.95 :[%s,%s]" % IC_95)
#IC - 95%:[ 174.22 , 176.18 ]
#Il y a un décalage de l'intervalle à cause du changement de moyenne
#mais surtout la largeur de l'intervalle diminue.


###################Exercice 3.4#########################

taille=[ 166, 170, 161, 167, 168, 169, 169,166, 163, 161, 162, 171, 169, 156, 168]

#1

#D'abord, la fonction st.sem renvoie sigma_n(x)/n^{1/2} comme illustré par les deux calculs suivants
st.sem(taille)
np.std(taille,ddof=1)/np.sqrt(len(taille))

#intervalle calculé directement 
st.t.interval(0.95,df=len(taille)-1,loc=np.mean(taille),scale=st.sem(taille))
#(163.38881498488192, 168.07785168178472)
#On arrondit en élargissant l'intervalle à [163.38, 168.08]


#2 
def StudentInterval(x,alpha=0.05):
    m=np.mean(x)
    s = np.std(x,ddof=1)
    l=len(x)
    delta = (st.t.ppf(1-alpha/2,df=l-1)*s) / np.sqrt(l)
    return (m-delta,m+delta)

#On retrouve la même valeur qu'avant:
StudentInterval(taille)
StudentInterval(taille,alpha=0.1)
###################Exercice 3.5#########################


Don=pan.read_csv("http://math.univ-lyon1.fr/~dabrowski/Donnees.csv",sep="\t")
Don=pan.read_csv("http://math.univ-lyon1.fr/homes-www/dabrowski/Donnees.csv",sep="\t")
for nom in Don.keys():
    globals()[nom] = Don[nom]
    
#1
th=taille[sexe=="H"]
tf=taille[sexe=="F"]
#vérification
pan.value_counts(sexe)
len(th)
len(tf)

#Il faut supposer que les échantillons suivent une loi normale

#2
st.ttest_1samp(th,popmean=176.6)
#p-valeur: 0.004276<0.01 donc on rejette
# l'hypothèse d'égalité à la moyenne française avec une Très forte présomption contre.
# La moyenne est significativement différente.

#Si l'hypothèse alternative était H_1 \mu_h>176,6
st.ttest_1samp(th,popmean=176.6,alternative="greater")
#on obtiendrait  pvalue=0.002<0.01 donc la moyenne serait afortiori très significativement supérieure

st.ttest_1sam
#3
st.ttest_1samp(tf,popmean=163.9,alternative="greater")
#pvalue=3.9068048962237294e-07<0.01 donc on rejette
# l'hypothèse d'égalité à la moyenne française avec une extraordinaire présomption contre.


#Option en Python 3.7
1-st.t.cdf(st.ttest_1samp(tf,popmean=163.9)[0],df=len(tf)-1)


st.ttest_1samp(tf,popmean=176.6,alternative="greater")
#pvalue=0.06 dans [0.05, 0.1] donc on ne peut pas rejetter l'hypothèse que la moyenne u_f soit égale à la moyenne 
#des tailles des hommes en France mais on a une faible présomption contre cette hypothèse.
#bien que la moyenne soit au dessus de cette valeur
np.mean(tf)#180.64
#Autrement dit, la moyenne u_f n'est pas significativement au dessus 176.6 mais presque (la p-valeur à 6% est proche de 5%) 
#et un échantillon de taille plus grande pourrait faire changer la conclusion

#4
table=pan.crosstab(tabac,ronfle)
st.chi2_contingency(table)
#Tout d'abord, la fonction renvoie la table des effectifs théoriques sous l'hypothèse d'indépendance:
#    array([[23.4, 12.6],
#        [41.6, 22.4]]))
 
#On vérifie que toutes les valeurs sont supérieures à 5, on peut effectuer le test.   

#On obtient p= 0.4065997565708874 >0.10 donc on a aucune présomption
# contre l'hypothèse nulle d'indépendance

#5
table=pan.crosstab(sexe,alcool)
st.chi2_contingency(table)
#Si on fait le test de base
#array([[10.5 ,  0.75,  1.5 ,  2.5 ,  3.5 ,  1.5 ,  0.5 ,  0.75,  2.  ,
#          1.  ,  0.25,  0.25],
#        [31.5 ,  2.25,  4.5 ,  7.5 , 10.5 ,  4.5 ,  1.5 ,  2.25,  6.  ,
#          3.  ,  0.75,  0.75]]))

#Les effectifs attendus sont biens inférieurs à 5, il faut regrouper des classes.

# On peut regrouper 1 à 4  et au dessus de 5

def RegroupeAlcool(x):
    if x==0: return("Aucun verre")
    elif (x>0 and x<5 ): return("Entre 1 et 4 verres")
    else: return("Plus de 5 verres")

Alcool=alcool.apply(RegroupeAlcool)
Alcool=Alcool.astype('category')

#On peut faire le test maintenant !
table=pan.crosstab(sexe,Alcool)
st.chi2_contingency(table)
# La table des effectifs théoriques a bien des valeurs au dessus de 5 à présent: 
# array([[10.5 ,  8.25,  6.25],
#        [31.5 , 24.75, 18.75]]))
 
#p=3.244020644336543e-08<0.01 donc on rejette l'hypothèse d'indépendance
#avec une très forte présomption contre 
   
###################Exercice 3.6#########################
from pydataset import data
iris=data("iris");y=iris["Petal.Length"]
x=iris["Sepal.Length"];l=len(x)

#1
st.pearsonr(x,y)
#p-valeur = 1.0386674194497525e-47<0.01 On rejette l'hypothèse de décorrélation avec une très forte présomption contre.
#et la corrélation est de 0.87

#2
import statsmodels.api as sm
X=np.column_stack((x,np.ones(l)))
res=sm.OLS(y,X).fit();print(res.summary())
#lm=sm.OLS(y,X);res=lm.fit();print(res.summary())

#76% de la variance de y est capturé par la régression linéaire
#les deux coefficients sont significativement non nuls
#la regresssion est y= 1.8584 x -7.1014

#3
prediction=res.get_prediction().summary_frame(alpha=0.05)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, "o", label="data")
ax.plot(x,  prediction["mean"], label="OLS",color="blue")#droite de régression
ax.plot(x, prediction["obs_ci_lower"], color="red")#borne inf de la prédiction
ax.plot(x, prediction["obs_ci_upper"], color="red")#borne sup  de la prédiction
#ax.plot(x, prediction["mean_ci_lower"], color="green")#Pas demandé, incertitude inférieur sur la droite
#ax.plot(x, prediction["mean_ci_upper"], color="green")#Pas demandé, incertitude supérieure sur la droite
ax.legend(loc="best")
fig.suptitle("Régression de la longueur des Pétales en fonction de celle des Sépales avec intervalle de prédiction (rouge)")

#On reprend la même commande en remplaçant l'échantillon par la nouvelle valeur de x=7.2
res.get_prediction(exog=[(7.2,1)]).summary_frame(alpha=0.05)
#ON obtient l'intervalle de prédiction à 95% [4.543332,8.015217]
