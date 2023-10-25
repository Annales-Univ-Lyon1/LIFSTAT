import pandas as pan #pour les donnÃ©es :gestion des data.frame
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.mstats as ms

pip install pydataset
from pydataset import data
pydata=data()

###################Exercice 1#########################

cars=data("cars")
data('cars', show_doc=True)

#1
np.mean(cars.speed)#moyenne
np.var(cars.speed)#variance empirique
np.var(cars.speed,ddof=1)#variance empirique non biaisée
np.std(cars.speed,ddof=1)#écart-type
np.median(cars.speed)#médiane

np.mean(cars.dist)#moyenne
np.var(cars.dist)#variance empirique
np.var(cars.dist,ddof=1)#variance empirique non biaisée
np.std(cars.dist,ddof=1)#écart-type
np.median(cars.dist)#médiane

#2
np.quantile(cars.dist, [0.25,0.5,0.75],interpolation="lower")
np.quantile(cars.speed, [0.25,0.5,0.75],interpolation="lower")


#3
cars.describe()

#           speed        dist
#count  50.000000   50.000000 Effectif
#mean   15.400000   42.980000 Moyenne
#std     5.287644   25.769377 écart-type 
#min     4.000000    2.000000
#25%    12.000000   26.000000 1er quartile
#50%    15.000000   36.000000 médiane
#75%    19.000000   56.000000 2ème quartile
#max    25.000000  120.000000

#On obtient tout en 1 ! ou presque

#4

fig, ax = plt.subplots()
ax.hist(cars.speed, color="blue")
ax.set_ylabel('Effectifs')
ax.set_xlabel('Vitesses')
ax.legend(frameon=False)
fig.suptitle("Histogramme de la variable speed")


###################Exercice 2#########################

#1
iris=data("iris")

#2

###############"Partie diagramme circulaire################
iris["Species"]=iris["Species"].astype("category")
plt.pie(iris["Species"].value_counts(),labels=iris["Species"].cat.categories)

###############"Partie générale################

iris.describe()
#      Sepal.Length  Sepal.Width  Petal.Length  Petal.Width
#count    150.000000   150.000000    150.000000   150.000000
#mean       5.843333     3.057333      3.758000     1.199333
#std        0.828066     0.435866      1.765298     0.762238
#min        4.300000     2.000000      1.000000     0.100000
#25%        5.100000     2.800000      1.600000     0.300000
#50%        5.800000     3.000000      4.350000     1.300000
#75%        6.400000     3.300000      5.100000     1.800000
#max        7.900000     4.400000      6.900000     2.500000

#Variance empirique
iris.var(ddof=0)
#Variance empirique non biaisée
iris.var(ddof=1)

###############"Partie par type d'espèce################

#Pour l'espèce Setosa, on trouve les moyennes avec describe 

iris[iris["Species"]=="setosa"].describe()

ou directement
iris[iris["Species"]=="setosa"].mean()

#Toutes les variances (non biaisées) par espèces
iris[iris["Species"]=="setosa"].var(ddof=1)#variances des setosa
iris[iris["Species"]=="versicolor"].var(ddof=1)
iris[iris["Species"]=="virginica"].var(ddof=1)

iris[iris["Species"]=="setosa"].describe()


#3
iris.boxplot(column="Sepal.Length",by="Species")
#Remarque : Cela distingue les espèces, mais ce n'est pas le trait le plus 
#distinctif: par exemple "Petal.Width"
iris.boxplot(column="Petal.Width",by="Species")
    

###################Exercice 3#########################


#1

x=np.array([1,8,5,1])

#2
y=np.concatenate(([0],1+2*np.array(range(5))))
y=np.concatenate(([0],[i for i in range(10) if i%2==1]))
y=np.concatenate(([0],range(1,10,2))) 


#3
y[2]
y[2:4]
y[-2]#7
y[y<=5]

#4
y[2*np.array(range(3))]
#ou 
y[2*np.arange(3)]
y[y>=1]

#5
y[1:]
y[y>0]#on triche vu que les valeurs y>0 correspondent à x>0



#6
X=np.repeat(x.reshape(1,4),25,axis=0).reshape(100)

#Attention: La commande suivante ne donne pas ce qu'on demande:
   
   ###################Exercice 4#########################


   #1 
   plt.plot(x,y)
   
   #cela renvoie un message sur la longueur différente des deux vecteurs.
   
   #2
 x=np.concatenate((x,[3,5]))

#3
fig,ax=plt.subplots()
ax.plot(x,y,'.')

#4
fig,ax=plt.subplots()
ax.plot(x,y,'.')
ax.plot(np.mean(x),np.mean(y),'r+')

#5
a,b=ms.linregress(x,y)[:2]
print("La droite de régression est y=%s x + %s" % (a,b))

fig,ax=plt.subplots()
ax.plot(x,y,'.')
ax.plot(np.mean(x),np.mean(y),'r+')
ax.axline(xy1=(0,b),slope=a,color="green")
fig.suptitle("Droite de régression de y en fonction de x")


#6
np.corrcoef(x,y)[0,1]#=-0.03873023356985952 c'est très proche de 0 
#donc l'approximation par la droite de régression n'est pas pertinente

#A la fin du cours, on verra un test qui précisera cette idée intuitive

   ###################Exercice 5#########################
#1
women=data("women")
x=women.height
y=women.weight
a,b=ms.linregress(x,y)[:2]
fig,ax=plt.subplots()
ax.set_xlim(min(x),max(x))
ax.set_ylim(100,180)
ax.plot(x,y,'.')
ax.axline(xy1=(0,b),slope=a,color="green")
fig.suptitle("Droite de régression du poids en fonction de la taille")

#le point de la courbe est la moyenne des tailles pour un poids donné, un point au dessus représente une personne beaucoup plus grande que la moyenne pour son poid.

   ################### Exercice 6 #########################
   Don=pan.read_csv("http://math.univ-lyon1.fr/~dabrowski/Donnees.csv",sep="\t")

   
Don=pan.read_csv("http://math.univ-lyon1.fr/homes-www/dabrowski/Donnees.csv",sep="\t")

print("Cles du tableau de donnees : %s" % Don.keys())
for nom in Don.keys():
    globals()[nom] = Don[nom]


#1
sexe=sexe.astype('category')
sexe.cat.categories=["Femme","Homme"]

ronfle=ronfle.astype('category')
ronfle.cat.categories=["Non-ronfleur","Ronfleur"]
tabac=tabac.astype('category')
tabac.cat.categories=["Non-fumeur","Fumeur"]

#typage des variables quantitatives continues
taille=taille.astype('float64')
poids=poids.astype('float64')
age=age.astype('float64')
#typage des variables quantitatives discrètes
alcool=alcool.astype('int64')



#2
df=pan.DataFrame({"age":age,"poids":poids,"taille":taille})
df.cov()
df.corr()
import seaborn as sns
sns.pairplot(df)

#Cela trace sur la diagonale l'histogramme de chaque variable, 
#et en dehors, les nuages de points 2 par deux.
#On voit la corrélation avec df.corr(), on trouve 0.926974 
#comme valeur max pour la corrélation de la taille et du poids
# C'est aussi visiblement les plus allignés sur les nuages de points
 

#3
a,b=ms.linregress(taille,poids)[:2]
fig,ax=plt.subplots()
ax.set_xlim(min(taille),max(taille))
ax.set_ylim(min(poids),max(poids))
ax.plot(taille,poids,'.')
ax.axline(xy1=(0,b),slope=a,color="red")
fig.suptitle("Droite de régression du poids en fonction de la taille")


# 4
df.boxplot()

#5 
pan.crosstab(ronfle,tabac,normalize=True)

#tabac         Non-fumeur  Fumeur
#ronfle                          
#Non-ronfleur          0.21      0.44
#Ronfleur              0.15      0.20
# le mode sont les fumeurs non-ronfleurs.

#6

alcool.hist(cumulative=True,histtype="step")

#alternative
from statsmodels.distributions.empirical_distribution import ECDF
ecdfA=ECDF(alcool)
plt.step(ecdfA.x,ecdfA.y)

ecdfP=ECDF(poids)
plt.step(ecdfP.x,ecdfP.y)
#Ce diagramme est presque une courbe et representer une interpolation linéaire, pour la cdf
# de la densité associée à un histogramme serait plus cohérent avec la représentation d'un histogramme
# pour les variables continues
