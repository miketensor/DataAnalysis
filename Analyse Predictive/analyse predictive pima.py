# -*- coding: utf-8 -*-

import os
os.chdir("... votre répertoire ...")

#utilisation de la librairie Pandas
#spécialisée - entres autres - dans la manipulation des données
import pandas
pima = pandas.read_table("pima.txt",sep="\t",header=0)

#dimensions
print(pima.shape)

#liste des colonnes
print(pima.columns)

#liste des colonnes et leurs types
print(pima.dtypes)

#transformation en matrice numpy
data = pima.as_matrix()

#### SCHEMA APPRENTISSAGE - TEST ####

#X matrice des var. explicatives
X = data[:,0:8]

#y vecteur de la var. à prédire
y = data[:,8]

#module model_selection
from sklearn import model_selection

#subdivision des données - test = 300, app = 768-test = 468
X_app,X_test,y_app,y_test = model_selection.train_test_split(X,y,test_size = 300,random_state=0)
print(X_app.shape,X_test.shape,y_app.shape,y_test.shape)


#à partir du module linear_model du package sklearn
#importer la classe LogisticRegression
from sklearn.linear_model import LogisticRegression

#création d'une instance de la classe
lr = LogisticRegression()

#exécution de l'instance sur les données d'apprentissage
#c.à-d. construction de f(.), le modèle prédictif
modele = lr.fit(X_app,y_app)

#les sorties sont très pauvres
#les coefficients
print(modele.coef_,modele.intercept_)

#prediction sur l'échantillon test
y_pred = modele.predict(X_test)

#importation de metrics
#utilisé pour les mesures de performances
from sklearn import metrics

#matrice de confusion
cm = metrics.confusion_matrix(y_test,y_pred)
print(cm)

#taux de succès
acc = metrics.accuracy_score(y_test,y_pred)
print(acc)

#taux d'erreur
err = 1.0 - acc
print(err)

#sensibilité
se = metrics.recall_score(y_test,y_pred,pos_label='positive')
print(se)

#écrire sa propre func. d'éval - ex. specificité
def specificity(y,y_hat):
    #matrice de confusion
    mc = metrics.confusion_matrix(y,y_hat)
    #negative est sur l'indice 0 dans la matrice
    import numpy
    res = mc[0,0]/numpy.sum(mc[0,:])
    #retour
    return res
#

#la rendre utilisable - transformation en objet scorer
specificite = metrics.make_scorer(specificity,greater_is_better=True)

#utilisation
sp = specificite(modele,X_test,y_test)
print(sp)

#### VALIDATION CROISEE ####

#importer la classe LogisticRegression)
from sklearn.linear_model import LogisticRegression

#création d'une instance de la classe
lr = LogisticRegression()

#exécution de l'instance sur la totalité des données (X,y)
modele_all = lr.fit(X,y)

#affichage
print(modele_all.coef_,modele_all.intercept_)

#utilisation du module cross_validation
from sklearn import cross_validation

#évaluation en validation croisée
succes = cross_validation.cross_val_score(lr,X,y,cv=10,scoring='accuracy')

#détail des itérations
print(succes)

#estimation du taux de succès en CV
print(succes.mean())


#### SCORING ####

#classe Régression Logistique
from sklearn.linear_model import LogisticRegression

#création d'une instance de la classe
lr = LogisticRegression()

#modélisation sur les données d'apprentissage
modele = lr.fit(X_app,y_app)

#calcul des probas d'affectation sur ech. test
probas = lr.predict_proba(X_test)

#score de 'presence'
score = probas[:,1]

#transf. en 0/1 de Y_test
pos = pandas.get_dummies(y_test).as_matrix()

#colonne de p
pos = pos[:,1]

#nombre total de positif
import numpy
npos = numpy.sum(pos)

#index pour tri selon le score croissant
index = numpy.argsort(score)

#inverser pour score décroissant
index = index[::-1]

#tri des individus (des valeurs 0/1)
sort_pos = pos[index]

#somme cumulée
cpos = numpy.cumsum(sort_pos)

#rappel
rappel = cpos/npos

#nb. obs ech.test
n = y_test.shape[0]

#taille de cible
taille = numpy.arange(start=1,stop=301,step=1)

#passer en pourcentage
taille = taille / n

#graphique
import matplotlib.pyplot as plt
plt.title('Courbe de gain')
plt.xlabel('Taille de cible')
plt.ylabel('Rappel')
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(taille,taille,marker='.',color='blue')
plt.scatter(taille,rappel,marker='.',color='red')
plt.show()


#### GRID SEARCH ####

#svm
from sklearn import svm

#par défaut un noyau RBF et C = 1.0
mvs = svm.SVC()

#modélisation
modele2 = mvs.fit(X_app,y_app)

#prédiction ech. test
y_pred2 = modele2.predict(X_test)

#matrice de confusion
print(metrics.confusion_matrix(y_test,y_pred2))

#succès en test
print(metrics.accuracy_score(y_test,y_pred2))

#recherche des combinaisons de paramétrage
parametres = [{'C':[0.1,1,10],'kernel':['rbf','linear']}]

#import de la classe
from sklearn import model_selection

#évaluation en validation croisée
#accuracy sera le critère à utiliser pour sélectionner la meilleure config.
grid = model_selection.GridSearchCV(estimator=mvs,param_grid=parametres,scoring='accuracy')

#lancer la recherche
#attention - gourmand en calculs
grille = grid.fit(X_app,y_app)

#résultat pour chaque combinaison
print(pandas.DataFrame.from_dict(grille.cv_results_).loc[:,["params","mean_test_score"]])

#meilleur paramétrage
print(grille.best_params_)

#meilleur performance
print(grille.best_score_)

#prédiction avec le meilleur modèle
y_pred3 = grille.predict(X_test)

#succès en test
print(metrics.accuracy_score(y_test,y_pred3))

#### SELECTION DE VARIABLES ####

#importer la classe LogisticRegression)
from sklearn.linear_model import LogisticRegression

#création d'une instance de la classe
lr = LogisticRegression()

#algorithme de sélection de var.
from sklearn.feature_selection import RFE
selecteur = RFE(estimator=lr)

#lancer la recherche
sol = selecteur.fit(X_app,y_app)

#nombre de var. sélectionnées
print(sol.n_features_)

#liste des variables sélectionnées
print(sol.support_)

#ordre de suppression
print(sol.ranking_)

#réduction de la base d'app.
X_new_app = X_app[:,sol.support_]
print(X_new_app.shape)

#construction du modèle
modele_sel = lr.fit(X_new_app,y_app)

#réduction de la base test
X_new_test = X_test[:,sol.support_]
print(X_new_test.shape)

#prédiction
y_pred_sel = modele_sel.predict(X_new_test)

#évaluation
print(metrics.accuracy_score(y_test,y_pred_sel))
