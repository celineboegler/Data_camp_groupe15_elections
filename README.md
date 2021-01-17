# RAMP Starting kit pour prédiction du taux d'abstention

*Auteurs : Ahmed Essakouri, Arnaud Venencie, Céline Boegler, Ludovic Figarella et Margot Herin*

Dans le cadre du cours Data Camp, au sein du Data Science Masters de l'Institut Polytechnique
de Paris, nous avons élaboré un projet en Data Science ayant pour objectif de prédire au mieux un aspect social, scientifique ou économique.

## Contexte de notre projet

En 2020, les élections municipales ont connu des taux d'abstention record, notamment au second tour avec une moyenne de 58.4% sur le territoire français. Le taux d'abstention a dépassé dans certaines villes les 75% comme par exemple à Roubaix (77,25 %).

Ces records ont été enregistrés dans le contexte particulier de la pandémie, contexte qui peut expliquer un détournement d'une partie de la population des questions civiques. <br>

Cependant, nous avons eu l'intuition que ces conditions extrêmes, provoquant l'accentuation des inégalités sociales en France, ont pu également accentuer les variations du taux d'abstention à travers le pays, et révéler les facteurs principaux d'un fort taux d'abstention. <br>

En effet, comme nous allons le voir par la suite, le taux d'abstention en France est très hétérogène et c'est une question sociale très complexe et primordiale que de comprendre quels sont les facteurs qui peuvent jouer sur ce taux. 
Nous avons rassemblé plusieurs bases de données contenant une gamme d'information socio-économique très large sur chaque ville afin de pouvoir considérer tous les facteurs possibles. <br>

Ce data challenge a donc pour objectif de **trouver les relations liant ces facteurs au taux d'abstention** par la construction d'un modèle prédictif adapté.

## Installation :


Ouvrez un terminal et installez la bibliothèque ramp-workflow (si ce n'est pas déjà fait) :

1. `$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git`

2. Suivez les instructions du [Wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit) pour les ramp-kit 


## Notebook

Pour commencer, ouvrez le notebook qui vous donnera des détails sur le challenge et quelques analyses des données.

Pour tester le "starting-kit", lancez

```shell
ramp_test_submission --submission starting_kit
```

Ou pour le faire en mode test :

```shell
ramp_test_submission --submission starting_kit --quick-test
```