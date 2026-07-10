# AgriculAI — Suivi de végétation par télédétection Sentinel-2

Pipeline complet et application interactive de calcul d'indices de
végétation à partir de données satellite Sentinel-2, avec acquisition
automatisée, zonage par machine learning (K-Means) et démonstration de
prédiction de rendement (XGBoost).

---

## Sommaire

1. Vue d'ensemble
2. Architecture du projet
3. Les fondamentaux scientifiques
4. toolkit.py — la boîte à outils
5. pipeline.py — traitement en ligne de commande
6. acquisition.py — récupération automatisée via l'API Copernicus
7. yield_prediction.py — démonstration de machine learning
8. app.py — l'application interactive
9. Installation et lancement
10. Limites connues et pistes d'amélioration

---

## Vue d'ensemble

Le projet répond à une question simple : comment suivre l'état de santé
d'une parcelle agricole à distance, sans s'y déplacer, à partir de données
satellite gratuites ?

Il s'articule en deux grandes capacités, réunies dans une application
Streamlit à deux onglets :

- Suivi satellite : calcul de 7 indices de végétation à partir de bandes
  Sentinel-2, zonage de la parcelle (par seuils manuels ou par clustering
  K-Means), et recherche de nouvelles zones par simple nom de ville via
  l'API Copernicus.
- Prédiction de rendement : démonstration d'un modèle XGBoost entraîné
  sur des données représentatives d'un cas réel, avec évaluation et
  importance des variables.

---

## Architecture du projet

```
AgriculAI/
├── toolkit.py            la boîte à outils : fonctions réutilisables
├── pipeline.py             assemble le toolkit en traitement batch
├── acquisition.py          récupère des images fraîches via l'API Copernicus
├── yield_prediction.py     démonstration de prédiction de rendement (ML)
├── app.py                   l'application interactive (Streamlit)
├── requirements.txt        dépendances Python
├── .gitignore                fichiers exclus du versionnement
├── SETUP.md                  guide de configuration de l'accès API
├── .streamlit/
│   └── secrets.toml         identifiants Copernicus (jamais versionné)
└── images/                   bandes Sentinel-2 d'exemple
```

Principe de conception : toolkit.py est une bibliothèque de fonctions
pures, sans état ni effet de bord caché. Les trois autres modules
(pipeline.py, acquisition.py, app.py) la consomment chacun pour un usage
différent — traitement automatisé, acquisition de données, interface
utilisateur — sans jamais dupliquer sa logique.

---

## Les fondamentaux scientifiques

### Le principe physique derrière les indices de végétation

Une plante en bonne santé absorbe fortement la lumière rouge (utilisée par
la chlorophylle pour la photosynthèse) et réfléchit fortement le proche
infrarouge (PIR), grâce à la structure interne de ses feuilles. Ce
contraste rouge/PIR diminue quand la plante est stressée, malade, ou en fin
de cycle — c'est ce contraste que tous les indices de végétation exploitent,
chacun avec une paire de bandes différente pour isoler un facteur précis
(vigueur générale, teneur en eau, statut azoté...).

### Les bandes Sentinel-2 utilisées

| Bande | Nom | Longueur d'onde | Résolution native |
|---|---|---|---|
| B02 | Bleu | ~490 nm | 10 m |
| B03 | Vert | ~560 nm | 10 m |
| B04 | Rouge | ~665 nm | 10 m |
| B05 | Red Edge | ~705 nm | 20 m |
| B08 | Proche infrarouge (PIR) | ~842 nm | 10 m |
| B11 | SWIR | ~1610 nm | 20 m |
| SCL | Scene Classification Layer | — | 20 m |

### Les 7 indices calculés

| Indice | Formule | Ce qu'il mesure |
|---|---|---|
| NDVI | (PIR-Rouge)/(PIR+Rouge) | Vigueur générale |
| NDRE | (PIR-RedEdge)/(PIR+RedEdge) | Vigueur, sans saturer sur forte densité |
| GNDVI | (PIR-Vert)/(PIR+Vert) | Statut azoté / chlorophylle |
| NDWI | (PIR-SWIR)/(PIR+SWIR) | Teneur en eau (nécessite le SWIR) |
| SAVI | (PIR-Rouge)/(PIR+Rouge+L) x (1+L) | Vigueur corrigée du sol nu |
| EVI | 2.5x(PIR-Rouge)/(PIR+6Rouge-7.5Bleu+1) | Vigueur, corrigée de l'atmosphère |
| MSAVI2 | variante auto-calibrée du SAVI | Vigueur corrigée du sol, sans paramètre à régler |

Point important : le NDWI est le seul indice qui ne peut pas être calculé
avec une constellation comme Planet, qui ne possède aucune bande SWIR —
c'est pourquoi ce projet s'appuie sur Sentinel-2, qui fournit gratuitement
l'ensemble des bandes nécessaires.

### Le masque qualité (bande SCL)

Plutôt que de deviner quels pixels sont des nuages ou des ombres, la bande
SCL fournie par l'ESA classe déjà chaque pixel dans une catégorie précise
(végétation, sol nu, eau, nuage à différents niveaux de confiance, ombre,
neige...). Le projet exclut systématiquement les catégories non fiables
avant tout calcul d'indice, garantissant des statistiques non faussées par
des pixels invalides.

---

## toolkit.py — la boîte à outils

Organisé en 8 sections thématiques.

### 1. Télédétection et indices

load_band, resample_band, ndvi, ndre, gndvi, ndwi, savi, evi, msavi2,
compute_all_indices, index_stats, classify_ndvi_thresholds.

Les 4 indices normalisés (ndvi, ndre, gndvi, ndwi) partagent la même
structure mathématique (a-b)/(a+b) — factoriser leur logique générale n'a
pas été retenu au profit de la lisibilité (chaque fonction porte un nom
explicite). Toutes les divisions sont protégées par np.errstate pour
éviter les avertissements sur les pixels où le dénominateur vaut zéro
(bordures, pixels non mesurés) : le résultat y reste NaN, ce qui est le
comportement voulu.

classify_ndvi_thresholds centralise la logique de zonage par seuils,
réutilisée à l'identique par pipeline.py et app.py, pour éviter toute
duplication de code.

### 2. Masque qualité (SCL)

SCL_CLASSES_TO_EXCLUDE, scl_mask, cloud_cover_ratio.

### 3. Agronomie

growing_degree_days, cumulative_gdd, simplified_water_balance,
nitrogen_dose_forecast_balance, classify_phenological_stage.

Cette section n'est pas encore branchée dans l'application — elle reste
disponible pour un futur enrichissement du suivi (affichage du stade
phénologique, du bilan hydrique...).

### 4. Séries temporelles

fill_time_gaps, smooth_savitzky_golay, extract_time_series_features,
decompose_trend_seasonality.

Non branchée non plus dans l'application actuelle, qui travaille sur une
seule date à la fois plutôt que sur une série temporelle complète.

### 5. Machine learning classique

temporal_split, train_xgboost_yield_model, feature_importance_table,
temporal_cross_validation.

C'est cette section qui alimente yield_prediction.py. Le point le plus
important : temporal_split sépare toujours l'entraînement et le test par
année, jamais aléatoirement — un split aléatoire sur des données agricoles
temporelles créerait une fuite d'information (le modèle pourrait deviner
une année à partir de données d'une autre année très proche dans le
temps), ce qui fausserait complètement l'évaluation.

### 6. Vision par ordinateur

excess_green_index, threshold_segmentation, count_objects,
vegetation_cover_ratio.

Pensée pour l'analyse d'images drone RGB classiques (sans bande
infrarouge), non branchée dans l'application actuelle.

### 7. Visualisation

plot_index_map, plotly_time_series_chart, dash_dashboard_skeleton.

plot_index_map calcule automatiquement une échelle de couleur adaptée aux
2e et 98e percentiles des données plutôt que d'utiliser une échelle fixe
-1/1, qui écraserait le contraste sur des indices dont les valeurs restent
proches de 0 (typiquement le NDWI).

### 8. Pipeline, configuration, logging

setup_logger, load_config, save_config, ProcessingPipeline, upload_to_s3.

ProcessingPipeline orchestre une suite d'étapes nommées avec logging à
chaque étape, sur le modèle simplifié d'un orchestrateur type AWS Step
Functions — chaque échec est journalisé avec le nom précis de l'étape
concernée avant d'être relancé.

---

## pipeline.py — traitement en ligne de commande

Assemble le toolkit en un pipeline exécutable via python pipeline.py, sans
avoir besoin d'un notebook. Cinq étapes enchaînées via ProcessingPipeline :

1. load_bands_step — charge les bandes depuis le disque
2. compute_indices_step — calcule les 7 indices
3. compute_stats_step — résume chaque indice en statistiques
4. generate_maps_step — exporte une carte PNG par indice
5. export_report_step — exporte un CSV et une carte de classification NDVI

Ce module travaille sur les images d'exemple incluses dans le dépôt, avec
des seuils de classification fixes — contrairement à app.py, il ne propose
pas d'interaction ni de choix de méthode de zonage.

---

## acquisition.py — récupération automatisée via l'API Copernicus

Permet de récupérer une image Sentinel-2 fraîche à partir d'un simple nom
de ville, sans passage par l'interface web de Copernicus Browser.

- geocode_city — convertit un nom de ville en coordonnées, via l'API
  gratuite Nominatim (OpenStreetMap)
- build_bbox — construit une petite zone rectangulaire autour du point
- _openeo_connection — connexion authentifiée par identifiants techniques
  fixes ("client credentials"), sans interaction utilisateur — indispensable
  pour une application publique déployée, où les visiteurs ne doivent
  jamais avoir à se connecter eux-mêmes
- fetch_latest_image — interroge l'API sur une plage de dates récentes,
  filtre par couverture nuageuse, et renvoie la date la plus récente
  exploitable

Nettoyage automatique : le fichier temporaire téléchargé (.nc) est
supprimé immédiatement après lecture, pour éviter que les fichiers
s'accumulent sur le disque du serveur à chaque nouvelle ville recherchée.

---

## yield_prediction.py — démonstration de machine learning

Ce module répond à un besoin précis : illustrer un vrai modèle supervisé de
bout en bout, sans dépendre d'un dataset externe à télécharger (pour que la
démonstration reste utilisable immédiatement, même dans une version
déployée publiquement).

- generate_synthetic_yield_dataset — génère des données réalistes mais
  synthétiques (8 années x 40 champs), où le rendement dépend du NDVI de
  pic, de l'aire sous la courbe, des degrés-jours de croissance cumulés et
  de la pluie cumulée, avec du bruit ajouté
- get_available_years — calcule dynamiquement la plage d'années
  disponibles, pour que le sélecteur d'année dans l'application reste
  toujours synchronisé avec les données réellement générées
- run_yield_prediction_demo — exécute le pipeline complet : génération,
  split temporel, entraînement XGBoost, évaluation, importance des variables

Transparence : les données utilisées ici sont explicitement synthétiques,
pas un vrai jeu de données agricole — la méthodologie (split temporel,
XGBoost, métriques d'évaluation) est en revanche directement transposable
à de vraies données de rendement.

---

## app.py — l'application interactive

Organisée en deux onglets Streamlit.

### Onglet "Suivi satellite"

- Choix entre une zone d'exemple (incluse dans le dépôt, chargement
  instantané) et une recherche par ville (appelle acquisition.py)
- Menu de sélection d'indice, avec un glossaire explicatif repliable
  (show_index_glossary)
- Curseur à double poignée pour les seuils de classification NDVI — les
  deux bornes ne peuvent jamais s'inverser, contrairement à deux curseurs
  indépendants
- Choix entre deux méthodes de zonage : seuils manuels, ou clustering
  K-Means (avec une explication pédagogique repliable, show_kmeans_explanation)

Le clustering K-Means (kmeans_zoning) combine trois indices (NDVI, NDWI,
GNDVI) pour regrouper les pixels en zones homogènes, sans seuil fixé à la
main — chaque pixel devient un point à 3 dimensions, et l'algorithme
découvre lui-même les regroupements naturels. Les clusters sont ensuite
triés par NDVI moyen croissant, pour garder une lecture cohérente quel que
soit le nombre de zones choisi.

### Onglet "Prédiction de rendement"

Appelle yield_prediction.py : sélection de l'année de test, entraînement
du modèle sur un clic, affichage des métriques (RMSE, MAE, R2), de
l'importance des variables, d'un graphique prédiction-vs-réalité, et du
détail des prédictions par champ.

### Gestion de l'état (st.session_state)

Comme Streamlit réexécute l'intégralité du script à chaque interaction
(changement de curseur, de menu...), les résultats d'une recherche par
ville ou d'un entraînement de modèle sont stockés dans st.session_state —
sans ça, changer simplement l'indice affiché effacerait le résultat de la
recherche précédente.

---

## Installation et lancement

### Environnement (conda recommandé)

```bash
conda create -n AgriculAI python=3.11
conda activate AgriculAI
conda install -c conda-forge rasterio geopandas numpy pandas scikit-learn xgboost scipy matplotlib streamlit openeo xarray netcdf4 requests
```

### Configuration de l'accès API (optionnel, pour la recherche par ville)

Voir SETUP.md pour créer un identifiant technique Copernicus. Sans cette
configuration, seul le mode "zone d'exemple" est utilisable.

### Lancement

```bash
streamlit run app.py
```

### Traitement batch (sans interface)

```bash
python pipeline.py
```

---

## Limites connues et pistes d'amélioration

- Les sections agronomie, séries temporelles et vision par ordinateur du
  toolkit sont codées mais pas encore branchées dans l'application
- Le zonage K-Means peut produire un bruit "poivre et sel" (pixels isolés
  mal classés), qu'un filtre médian post-traitement réduirait
- Le choix du nombre de clusters reste manuel ; une méthode du coude
  objectiverait ce choix
- Le module de prédiction de rendement utilise des données synthétiques,
  pas un vrai historique de rendement
- pyarrow (dépendance de certains widgets Streamlit comme st.table et
  st.bar_chart) peut poser des problèmes d'installation sous Windows —
  l'application privilégie des graphiques matplotlib pour rester robuste
  à cet égard
