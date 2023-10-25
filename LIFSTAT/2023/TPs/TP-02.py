#!/usr/bin/env python
# coding: utf-8

# ![logo](/images/logoUCB.png)
# # Licence L2, Université  Lyon 1 
# # [Statistiques pour l'informatique]

# ## Consignes pour les étudiants
# 
# 1. **Exécution**
# 
# **Exécuter toujours tout le notebook depuis le début** (en appuyant sur le bouton **`Executer`** ou **Run**). En effet un notebook est un programme et pas une simple page HTML qui doit s'exécuter dans l'ordre du début à la fin.
# 
# 2. **En cas de problème**
# Tout d'abord, redémarrez le noyau (dans la barre de menus, sélectionnez **`Noyau -> Redémarer`** ou  **`Kernel -> Restart`**),
# puis exécutez toutes les cellules (dans la barre de menus, sélectionnez **`Cellule -> Executer tout`** ou **`Cellule -> Run All`**).

# In[ ]:


from validation.validation import info_etudiant
NOM, PRENOM, NUMERO_ETUDIANT = info_etudiant()
print("Etudiant {} {}  id={}".format(NOM,PRENOM,NUMERO_ETUDIANT))


# <hr>

# # Fonctionnement des Travaux Pratiques

# Compléter le code après la ligne <code> # YOUR CODE GOES HERE </code>.  
# Une fois votre code écrit, supprimer la ligne <code> raise NotImplementedError() </code> avant d'exécuter.
# 
# Lien vers le cours : [site](https://licence-math.univ-lyon1.fr/doku.php?id=a23:s3_stats_pour_info:page)
# 
# Dans les cellules suivantes une explication des tests est donnée:

# In[ ]:





# Qui a gagné le tour de France de 2023 ?
# - 'A': Jonas Vingegaard 
# - 'B': Tadej Pogačar 
# - 'C': Thibaut Pinot 

# Quand la réponse est juste, aucun message d'erreur ne s'affiche.

# In[ ]:


ReponseQuestionTest = 'A'
assert(ReponseQuestionTest=='A')


# Quand la réponse est fausse, un message d'erreur s'affiche.

# In[ ]:


ReponseQuestionTest = 'B'
assert(ReponseQuestionTest=='A')


# <div class="alert alert-block alert-danger">
# <b>Attention:</b> Il faut répondre 'A' ou 'B' ou 'C'.</div>

# In[ ]:


ReponseQuestionTest = A
assert(ReponseQuestionTest=='A')


# <hr>

# <hr>

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats.mstats as ms
from pydataset import data


# # Exercice TP2.1
# On import les données à l'aide de la commande suivante

# In[ ]:


data('cars',show_doc=True)
cars=data('cars')


# 1. Calculer la moyenne, variance empirique, variance empirique non-biaisée, médiane, écart-type des variables <code>speed</code> et <code>dist</code>.

# In[ ]:


# Question 1
# YOUR CODE HERE
raise NotImplementedError()


# 2. Calculer les 3 quartiles selon la définition du cours

# In[ ]:


# Question 2
# YOUR CODE HERE
raise NotImplementedError()


# 3. Que fait la fonction <code>cars.describe()</code>?

# In[ ]:


# Question 3
# YOUR CODE HERE
raise NotImplementedError()


# 4. Selon le type de variables à notre disposition, quelle représentation graphique suggérez-vous ?    
# Tracer l’histogramme de la variable <code>speed</code>. Ajouter un titre, et des étiquettes aux axes $x$ et $y$.

# In[ ]:


# Question 4
# YOUR CODE HERE
raise NotImplementedError()


# # Exercice TP2.2

# Représentation graphique d’une variable qualitative.
# 1. Charger le jeu de données <code>iris</code>.

# In[ ]:


# Question 1
# YOUR CODE HERE
raise NotImplementedError()


# 2. Tracer le diagramme circulaire pour la variable qualitative.    
# Calculer la moyenne, variance empirique, variance non-biaisée, le minimum et le maximum pour les variables quantitatives. 
# Calculer la moyenne et la variance non-biaisée par type de Species. 
# Que remarquez vous ? 
# Quelle est l’autre façon de représenter une variable qualitative ?

# In[ ]:


# Question 2

# YOUR CODE HERE
raise NotImplementedError()


# 3. Tracer la boite à moustache de <code>Sepal.Length</code> en fonction de <code>Species</code>. Quel est le but de cette
# représentation ?

# In[ ]:


# Question 3
# YOUR CODE HERE
raise NotImplementedError()


# # Exercice TP2.3

# Rappels sur les vecteurs de <code>numpy</code>.
# 1. Créer le vecteur $x=(1,8,5,1)$ grâce à la commande code <code>np.array</code>

# In[ ]:


# Question 1
x=
# YOUR CODE HERE
raise NotImplementedError()


# 2. Créer le vecteur $y=(0,1,3,5,7,9)$ en utilisant <code>np.array</code>,<code>range</code> et <code>np.concatenate</code>

# In[ ]:


# Question 2
y = 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert(sum(y==np.array([0,1,3,5,7,9])))


# 3. Etudier les résultats des commandes <code>y[4]</code>, <code>y[2:4]</code>, <code>y[-2]</code>, et <code>y[y<=5]</code>

# In[ ]:


# Question 3 
# YOUR CODE HERE
raise NotImplementedError()


# 4. Extraire les éléments en position paire de $y$. Extraire les éléments plus grands que 1 de $y$.

# In[ ]:


# Question 4
y1 = 
y2 = 
# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert(sum(y1==np.array([0,3,7])))
assert(sum(y2==np.array([1,3,5,7,9])))


# 5. Conserver tous les élements de $y$, sauf le premier.

# In[ ]:


# Question 5
# YOUR CODE HERE
raise NotImplementedError()


# 6. A l’aide de les commandes <code>np.repeat()</code> et <code>np.reshape()</code>, créer un vecteur $X$ de taille 100 obtenu
# en mettant bout à bout 25 copies de $x$. (Donc, $X$ commence ainsi $X = (1, 8, 5, 1, 1, 8, 5, 1, \ldots)$)

# In[ ]:


# Question 6
X=

# YOUR CODE HERE
raise NotImplementedError()


# In[ ]:


assert(X[22]==5)
assert(X[47]==1)
assert(X[62]==5)
assert(X[80]==1)


# # Exercice TP2.4

# On considère à nouveau les vecteurs $x=(1,8,5,1)$ et $y=(0,1,3,5,7,9)$.
# 1. Pourquoi la commande <code>plt.plot(x,y)</code> retourne-t-elle une erreur?

# In[ ]:


# Question 1
# YOUR CODE HERE
raise NotImplementedError()


# 2. Ajouter $(3,5)$ au vecteur $x$ et représnter le nuage des points $(x_i,y_i)$.

# In[ ]:


# Question 2
# YOUR CODE HERE
raise NotImplementedError()


# 3. Ajouter le point $(\bar{x},\bar{y})$ en rouge en utilisant la commande <code>plt.plot</code>.

# In[ ]:


# Question 3
# YOUR CODE HERE
raise NotImplementedError()


# 4. Ajouter la droite de régression en utilisant <code>plt.axline</code> et <code>ms.linregress</code>

# In[ ]:


# Question 4
# YOUR CODE HERE
raise NotImplementedError()


# 5. Calculer la corrélation empirique $\text{cor}(x,y)$ pour décider si une approximation par une droite est raisonnable.

# In[ ]:


# Question 5
# YOUR CODE HERE
raise NotImplementedError()


# # Exercice TP2.5

# 1. Charger le jeu de données <code>women</code>. Représenter les deux variables "taille" et "poids" par un nuage de points, avec la droite de régression du poids en fonction de la taille

# In[ ]:


# Question 1

# YOUR CODE HERE
raise NotImplementedError()


# 2. Discuter si cette approximation est raisonnable. Comment interprète-t-on un point qui se trouve nettement au-dessus/au-dessous de la droite de régression

# # Exercice TP2.6

# Un fichier "Donnees.csv" est présent dans le dossier. Importer ce fichier avec la function <code>pd.read_csv("fichier.csv",sep="\t") </code>.
# Ces données correspondent à l’âge, au poids, à la taille, à la consommation hebdomadaire d’alcool (en nombre de verres bus), au sexe, au ronflement et au tabagisme, d’un échantillon de 100 personnes. 
#     
# 
# 1. Commencer par identifier les variables qualitatives nominales, ordinales et quantitative discrètes, continues. Typer les données qualitatives correctement en Python (on pourra utiliser les commandes <code>astype</code>) et renommer les niveaux avec la variable <code>cat.categories</code>.

# In[ ]:


# Question 1
# Récupération du fichier et création d'un DataFrame
Donnees= pd.read_csv("Donnees.csv",sep="\t") 

# Création des séries pour chaque variable
for nom in Donnees.keys():
    globals()[nom] = Donnees[nom]

    # YOUR CODE HERE
    raise NotImplementedError()


# 2. Calculer la corrélation entre poids et taille.   
# Que font les commandes suivantes ?   
# Quelles sont les variables continues les plus corrélées entre elles ?
# 
# <p style="background:#F3F3F3"> <code style="background:#F3F3F3;"># Code à recopier 
# import seaborn as sns 
# df=pd.DataFrame({"age":Donnees['age'],
#                   "poids":Donnees['poids'],
#                   "taille":Donnees['taille']})
# print('Premier print :\n',df.cov(),'\n'); 
# print('Deuxième print :\n',df.corr(),'\n')
# sns.pairplot(df);
# </code> </p>

# In[ ]:


# Question 2
# YOUR CODE HERE
raise NotImplementedError()


# 3. Tracer un nuage de points du **poids** en fonction de la **taille.**              
# Calculer la droite de régression de ces deux variables et l’ajouter en rouge au nuage de points.    
# Discuter si cette approximation est raisonnable. Avez-vous un commentaire sur cet échantillon de données ?

# In[ ]:


# Question 3

# YOUR CODE HERE
raise NotImplementedError()


# 4. Tracer sur un même graphique les diagrammes à moustaches de **age**, **poids** et **taille**.

# In[ ]:


# Question 4

# YOUR CODE HERE
raise NotImplementedError()


# 5. Calculer la table de contingence des fréquences de **ronfle** et **tabac.**      
# Quel est le mode du couple **(tabac,ronfle)** ?

# In[ ]:


# Question 5

# YOUR CODE HERE
raise NotImplementedError()



# 6. Tracer les fonctions de répartition empirique de **alcool** et **poids.**            
# Le diagramme en escalier vous paraît-il pertinent pour les deux?    
# Indication : <code>from statsmodels.distributions .empirical_distribution import ECDF </code>

# In[ ]:


# Question 6 

# YOUR CODE HERE
raise NotImplementedError()


