"""
TOOLKIT — IA appliquée à l'agriculture de précision

Organisé par thème :
  1. Télédétection et indices de végétation (Sentinel-2 / Planet)
  2. Masque qualité / nuages (SCL)
  3. Agronomie (phénologie, GDD, bilan hydrique)
  4. Séries temporelles (lissage, gap-filling, extraction de features)
  5. Machine learning classique (feature engineering, XGBoost, évaluation)
  6. Traitement d'image / vision par ordinateur (drone, détection)
  7. Visualisation (restitution, dashboard)
  8. Pipeline / config / logging
"""

import os
import numpy as np
import pandas as pd


# 1. télédétection et indices de végétation

import rasterio
from rasterio.enums import Resampling


def load_band(path, valid_mask=None, scale_factor=10000):
    """
    charge une bande sentinel-2/planet, filtre les pixels invalides selon un
    masque, et renvoie la réflectance (0-1) plutôt que les entiers bruts.

    exemple :
        red, profile = load_band("B04.tiff")
        print(red.min(), red.max())  -> valeurs entre 0 et ~1, pas 0-10000
    """
    with rasterio.open(path) as src:
        raw = src.read(1).astype(float)
        profile = src.profile
    if valid_mask is not None:
        raw = np.where(valid_mask, raw, np.nan)
    return raw / scale_factor, profile


def resample_band(path, target_shape, resampling=Resampling.bilinear, scale_factor=10000):
    """
    recharge une bande à résolution différente (ex: 20m) directement sur la
    grille cible (ex: 10m), pour pouvoir la combiner avec d'autres bandes.

    exemple :
        red, _ = load_band("B04_10m.tiff")           -> forme (852, 1250)
        swir = resample_band("B11_20m.tiff", target_shape=red.shape)
        # swir a maintenant exactement la même forme que red
    """
    with rasterio.open(path) as src:
        data = src.read(1, out_shape=target_shape, resampling=resampling).astype(float)
    return data / scale_factor


def ndvi(nir, red):
    """
    normalized difference vegetation index. mesure la vigueur générale de la
    végétation via le contraste absorption du rouge / réflexion du proche
    infrarouge. plage théorique [-1, 1].

    exemple : nir=0.48, red=0.06 -> ndvi environ 0.78 (végétation dense et saine)
    """
    # np.errstate évite les warnings "invalid value encountered in divide"
    # sur les pixels où nir+red=0 (bordures, pixels non mesurés) ; le
    # résultat reste NaN à ces endroits, c'est voulu.
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - red) / (nir + red)


def ndre(nir, red_edge):
    """
    red edge ndvi. même principe que le ndvi mais avec la bande "red edge" :
    reste discriminant sur une végétation déjà dense, là où le ndvi classique
    sature (n'augmente quasiment plus).

    exemple : nir=0.50, red_edge=0.20 -> ndre environ 0.43, alors que le
    ndvi de la même parcelle serait déjà saturé autour de 0.85.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - red_edge) / (nir + red_edge)


def gndvi(nir, green):
    """
    green ndvi. utilise le vert plutôt que le rouge, plus sensible à la
    concentration en chlorophylle sur la durée — utile pour suivre un statut
    azoté (l'azote est un composant essentiel de la chlorophylle).

    exemple : nir=0.45, green=0.11 -> gndvi environ 0.61
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - green) / (nir + green)


def ndwi(nir, swir):
    """
    water index (gao). ne mesure pas la vigueur mais la teneur en eau de la
    plante, via le contraste pir/swir — nécessite une bande swir
    (disponible sur sentinel-2, absente de planet).

    exemple : nir=0.44, swir=0.28 -> ndwi environ 0.22, signe de stress
    hydrique débutant, même si le ndvi de la même parcelle reste normal.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - swir) / (nir + swir)


def savi(nir, red, soil_factor=0.5):
    """
    soil-adjusted vegetation index. corrige la contribution du sol nu visible
    entre les jeunes plants — utile en tout début de cycle (levée, tallage),
    quand le ndvi classique serait faussé par le sol encore bien visible.

    exemple : nir=0.22, red=0.12, soil_factor=0.5 -> savi environ 0.22,
    plus représentatif de la vraie faible densité végétale qu'un ndvi de 0.29.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((nir - red) / (nir + red + soil_factor)) * (1 + soil_factor)


def evi(nir, red, blue):
    """
    enhanced vegetation index. corrige la saturation sur forte biomasse et
    les perturbations atmosphériques (le bleu sert de sonde à la diffusion
    des aérosols). attention : formule non bornée à [-1, 1], à clipper avant usage.

    exemple : nir=0.50, red=0.04, blue=0.05 -> evi environ 3.07 (valeur
    brute, à recadrer en production, ex: np.clip(valeur, -1, 2.5)).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


def msavi2(nir, red):
    """
    modified savi v2. évolution du savi qui calcule elle-même son propre
    facteur de correction du sol, sans avoir à deviner un paramètre à la main.
    """
    with np.errstate(invalid="ignore"):
        inner = (2 * nir + 1) ** 2 - 8 * (nir - red)
        inner = np.clip(inner, 0, None)  # évite une racine négative sur des pixels bruités
        return (2 * nir + 1 - np.sqrt(inner)) / 2


def compute_all_indices(bands: dict) -> dict:
    """
    calcule les 7 indices d'un coup à partir d'un dictionnaire de bandes.
    attend les clés B02, B03, B04, B05, B08, B11 (réflectance 0-1).

    exemple :
        bands = {"B02": blue, "B03": green, "B04": red, "B05": red_edge,
                 "B08": nir, "B11": swir}
        indices = compute_all_indices(bands)
        indices["NDVI"]  -> tableau 2D du ndvi
    """
    return {
        "NDVI": ndvi(bands["B08"], bands["B04"]),
        "NDRE": ndre(bands["B08"], bands["B05"]),
        "GNDVI": gndvi(bands["B08"], bands["B03"]),
        "NDWI": ndwi(bands["B08"], bands["B11"]),
        "SAVI": savi(bands["B08"], bands["B04"]),
        "EVI": np.clip(evi(bands["B08"], bands["B04"], bands["B02"]), -1, 2.5),
        "MSAVI2": msavi2(bands["B08"], bands["B04"]),
    }


def index_stats(array) -> dict:
    """
    résumé statistique robuste aux NaN — utile pour un rapport ou une carte
    de synthèse. utilise les variantes np.nan... pour ignorer automatiquement
    les pixels filtrés en amont, plutôt que de renvoyer NaN pour tout le résumé.

    exemple : index_stats(ndvi_array) -> {"mean": 0.68, "median": 0.71, ...}
    """
    valid_count = np.sum(~np.isnan(array))
    if valid_count == 0:
        # évite le warning "Mean of empty slice" si tous les pixels sont NaN
        # (zone entièrement masquée) plutôt que de laisser numpy paniquer
        return {"mean": np.nan, "median": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan, "valid_pct": 0.0}
    with np.errstate(invalid="ignore"):
        return {
            "mean": round(float(np.nanmean(array)), 3),
            "median": round(float(np.nanmedian(array)), 3),
            "std": round(float(np.nanstd(array)), 3),
            "min": round(float(np.nanmin(array)), 3),
            "max": round(float(np.nanmax(array)), 3),
            "valid_pct": round(100 * valid_count / array.size, 1),
        }


def classify_ndvi_thresholds(ndvi_array, soil_threshold=0.2, dense_threshold=0.5):
    """
    classe une carte ndvi en 4 zones (eau / sol nu / végétation faible /
    végétation dense) selon deux seuils. factorisée ici pour éviter de
    dupliquer cette logique dans pipeline.py et app.py — un seul endroit
    à corriger si la règle de classification doit changer un jour.

    exemple :
        classes = classify_ndvi_thresholds(ndvi_array, 0.2, 0.5)
        # classes vaut 0 (eau), 1 (sol nu), 2 (faible) ou 3 (dense) par pixel
    """
    classes = np.full(ndvi_array.shape, np.nan)
    classes = np.where(ndvi_array < 0, 0, classes)
    classes = np.where((ndvi_array >= 0) & (ndvi_array < soil_threshold), 1, classes)
    classes = np.where((ndvi_array >= soil_threshold) & (ndvi_array < dense_threshold), 2, classes)
    classes = np.where(ndvi_array >= dense_threshold, 3, classes)
    return classes


# 2. masque qualité / nuages (SCL — scene classification layer)

# catégories à exclure : nodata, pixel défectueux, ombres, nuages à
# différents niveaux de confiance, cirrus, neige.
SCL_CLASSES_TO_EXCLUDE = [0, 1, 2, 3, 7, 8, 9, 10, 11]


def scl_mask(scl_path, classes_to_exclude=SCL_CLASSES_TO_EXCLUDE):
    """
    construit un masque booléen (True = pixel valide) à partir de la bande
    scl officielle de l'esa — plus fiable qu'un simple filtre sur une valeur
    numérique suspecte, puisque chaque pixel est déjà classé par un vrai
    algorithme (sen2cor).

    exemple :
        mask = scl_mask("SCL.tiff")
        red_clean = np.where(mask, red_raw, np.nan)
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    return ~np.isin(scl, classes_to_exclude)


def cloud_cover_ratio(scl_path, cloud_classes=(3, 7, 8, 9, 10)):
    """
    calcule le vrai taux de nuages sur la zone précise étudiée — pas la
    scène sentinel-2 entière (100x100km) comme l'affiche copernicus browser.

    exemple : cloud_cover_ratio("SCL.tiff") -> 4.2 (soit 4.2% de nuages
    sur cette parcelle précise, indépendamment du taux affiché globalement)
    """
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
    return round(100 * np.isin(scl, cloud_classes).sum() / scl.size, 2)


# 3. agronomie (phénologie, GDD, bilan hydrique, bilan azoté)

def growing_degree_days(t_max, t_min, t_base=0):
    """
    GDD journalier : recale le suivi sur l'horloge biologique de la plante
    plutôt que sur le calendrier civil.

    exemple : t_max=18, t_min=8, t_base=0 -> GDD = (18+8)/2 - 0 = 13
    """
    return np.maximum(((t_max + t_min) / 2) - t_base, 0)


def cumulative_gdd(weather_df: pd.DataFrame, tmax_col="t_max", tmin_col="t_min", t_base=0) -> pd.Series:
    """
    cumul de GDD jour après jour sur une série météo — deux parcelles
    semées à la même date peuvent être à des stades différents selon le
    climat vécu.
    """
    daily_gdd = growing_degree_days(weather_df[tmax_col], weather_df[tmin_col], t_base)
    return daily_gdd.cumsum()


def simplified_water_balance(rain_mm: pd.Series, etp_mm: pd.Series, max_soil_capacity_mm=100, initial_stock=None):
    """
    bilan hydrique jour par jour : stock = stock précédent + pluie - etp,
    borné entre 0 et la réserve utile maximale du sol.

    note technique : calcul volontairement fait en boucle (pas vectorisé),
    car le stock d'un jour dépend directement de celui de la veille.
    """
    stock = initial_stock if initial_stock is not None else max_soil_capacity_mm
    results = []
    for rain, etp in zip(rain_mm, etp_mm):
        stock = np.clip(stock + rain - etp, 0, max_soil_capacity_mm)
        results.append(stock)
    return pd.Series(results, index=rain_mm.index)


def nitrogen_dose_forecast_balance(crop_needs_kg_ha, soil_supply_kg_ha, residual_n_kg_ha):
    """
    dose d'azote à apporter selon la méthode du bilan prévisionnel :
    dose = besoins de la culture - fourniture du sol - reliquat mesuré.
    le max(..., 0) évite de recommander une dose négative, absurde agronomiquement.

    exemple : besoins=180, fourniture=40, reliquat=30 -> dose = 110 kg/ha
    """
    return max(crop_needs_kg_ha - soil_supply_kg_ha - residual_n_kg_ha, 0)


def classify_phenological_stage(cumulative_gdd_value, thresholds: dict):
    """
    renvoie le stade phénologique correspondant à un GDD cumulé donné.

    exemple :
        thresholds = {"emergence": 80, "tillering": 250, "flowering": 900}
        classify_phenological_stage(300, thresholds) -> "tillering"
    """
    stage = "before_emergence"
    for stage_name, threshold in sorted(thresholds.items(), key=lambda x: x[1]):
        if cumulative_gdd_value >= threshold:
            stage = stage_name
    return stage


# 4. séries temporelles (lissage, gap-filling, extraction de features)

from scipy.signal import savgol_filter


def fill_time_gaps(dates: pd.Series, values: pd.Series) -> pd.Series:
    """
    interpolation linéaire des valeurs manquantes (trous nuageux) sur une
    série temporelle — comble un trou entre deux observations valides voisines.
    """
    return values.interpolate(method="linear", limit_direction="both")


def smooth_savitzky_golay(values: np.ndarray, window=7, polyorder=2):
    """
    lisse une série temporelle en préservant la forme des vrais pics —
    contrairement à une moyenne mobile simple, qui aurait tendance à
    aplatir un sommet de croissance réel.
    """
    window = window if window % 2 == 1 else window + 1  # la fenêtre doit être impaire
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    return savgol_filter(values, window_length=window, polyorder=polyorder)


def extract_time_series_features(dates: pd.Series, ndvi_series: pd.Series) -> dict:
    """
    résume une courbe ndvi entière en une poignée de features exploitables
    par du ML classique (XGBoost), en alternative à donner la séquence
    brute à un LSTM.

    exemple : une saison avec un pic à 0.82 le jour 75 donnera par exemple
    {"ndvi_peak": 0.82, "peak_day": 75, "area_under_curve": 48.3, ...}
    """
    days = (pd.to_datetime(dates) - pd.to_datetime(dates).min()).dt.days.values
    values = ndvi_series.values

    if np.all(np.isnan(values)):
        # série entièrement invalide (ex: toutes les dates couvertes de
        # nuages) : on renvoie des NaN plutôt que de laisser np.nanargmax
        # lever une ValueError
        return {
            "ndvi_peak": np.nan, "peak_day": np.nan, "ndvi_season_mean": np.nan,
            "area_under_curve": np.nan, "rising_slope": np.nan, "falling_slope": np.nan,
        }

    peak_idx = np.nanargmax(values)
    area_under_curve = np.trapz(np.nan_to_num(values), days)  # proxy de biomasse cumulée

    return {
        "ndvi_peak": round(float(values[peak_idx]), 3),
        "peak_day": int(days[peak_idx]),
        "ndvi_season_mean": round(float(np.nanmean(values)), 3),
        "area_under_curve": round(float(area_under_curve), 1),
        "rising_slope": round(float((values[peak_idx] - values[0]) / max(days[peak_idx] - days[0], 1)), 4),
        "falling_slope": round(
            float((values[-1] - values[peak_idx]) / max(days[-1] - days[peak_idx], 1)), 4
        ) if peak_idx < len(values) - 1 else 0.0,
    }


def decompose_trend_seasonality(series: pd.Series, period=365):
    """décomposition additive tendance/saisonnalité/résidu — nécessite statsmodels."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    return seasonal_decompose(series, model="additive", period=period)


# 5. machine learning classique (feature engineering, XGBoost, évaluation)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb


def temporal_split(df: pd.DataFrame, year_col: str, test_years: list):
    """
    split train/test cohérent dans le temps — jamais de split aléatoire sur
    des données temporelles agricoles, sous peine de fuite d'information du
    futur vers le passé.

    exemple : temporal_split(df, "year", test_years=[2024])
    -> entraîne sur toutes les années sauf 2024, teste uniquement sur 2024.
    """
    train = df[~df[year_col].isin(test_years)]
    test = df[df[year_col].isin(test_years)]
    return train, test


def train_xgboost_yield_model(X_train, y_train, X_test, y_test, **xgb_params):
    """
    entraîne un XGBoost de régression et renvoie le modèle ainsi que ses
    métriques d'évaluation (RMSE, MAE, R2).
    """
    default_params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    default_params.update(xgb_params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 3),
        "R2": round(float(r2_score(y_test, y_pred)), 3),
    }
    return model, metrics, y_pred


def feature_importance_table(model, feature_names) -> pd.DataFrame:
    """
    transforme la sortie brute d'un modèle (un simple tableau de nombres) en
    un dataframe trié et nommé — prêt à être montré tel quel pour vulgariser.
    """
    return pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def temporal_cross_validation(df, year_col, feature_cols, target_col, **xgb_params):
    """
    validation croisée "leave-one-year-out" : chaque année sert tour à tour
    de test, pour vérifier que le modèle généralise bien d'une année sur l'autre.
    """
    years = sorted(df[year_col].unique())
    results = []
    for test_year in years:
        train, test = temporal_split(df, year_col, [test_year])
        if len(test) == 0 or len(train) == 0:
            continue
        _, metrics, _ = train_xgboost_yield_model(
            train[feature_cols], train[target_col], test[feature_cols], test[target_col], **xgb_params
        )
        metrics["test_year"] = test_year
        results.append(metrics)
    return pd.DataFrame(results)


# 6. traitement d'image / vision par ordinateur (drone, détection adventices)

from scipy import ndimage


def excess_green_index(rgb_image: np.ndarray) -> np.ndarray:
    """
    ExG = 2*vert - rouge - bleu. segmente végétation/sol sur une simple
    image RGB de drone, sans avoir besoin d'une bande infrarouge.
    """
    r, g, b = rgb_image[..., 0].astype(float), rgb_image[..., 1].astype(float), rgb_image[..., 2].astype(float)
    return 2 * g - r - b


def threshold_segmentation(index_array: np.ndarray, threshold=None) -> np.ndarray:
    """
    seuillage simple. si threshold=None, calcule automatiquement le seuil
    optimal via la méthode d'otsu (à partir de l'histogramme de l'image).
    """
    if threshold is None:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(index_array)
    return index_array > threshold


def count_objects(binary_mask: np.ndarray, min_size_px=5) -> dict:
    """
    compte les objets connectés d'un masque binaire (ex: comptage de plants
    ou d'adventices individuels sur une image drone haute résolution).
    nettoie d'abord le bruit isolé avant de regrouper les pixels connectés.
    """
    cleaned_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3, 3)))
    labels, n_objects = ndimage.label(cleaned_mask)
    sizes = ndimage.sum(cleaned_mask, labels, range(1, n_objects + 1))

    # la taille moyenne doit être calculée uniquement sur les objets qui
    # passent le filtre min_size_px, sinon elle reste incohérente avec le
    # nombre d'objets renvoyé (objects_detected)
    valid_sizes = sizes[sizes >= min_size_px]

    return {
        "objects_detected": int(len(valid_sizes)),
        "average_size_px": round(float(np.mean(valid_sizes)), 1) if len(valid_sizes) else 0,
    }


def vegetation_cover_ratio(binary_mask: np.ndarray) -> float:
    """pourcentage de la zone couverte par de la végétation détectée — proxy de taux de levée."""
    return round(100 * binary_mask.sum() / binary_mask.size, 2)


# 7. visualisation (restitution, dashboard)

import matplotlib.pyplot as plt


def plot_index_map(array, title="Index", cmap="RdYlGn", vmin=None, vmax=None, save_path=None):
    """
    affiche/exporte une carte d'indice avec échelle de couleur adaptative
    (2e/98e percentile) si vmin/vmax ne sont pas fournis — évite qu'une
    échelle fixe -1/1 écrase le contraste réel quand les valeurs restent
    proches de 0 (typiquement le cas du ndwi).
    """
    if vmin is None:
        vmin = np.nanpercentile(array, 2)
    if vmax is None:
        vmax = np.nanpercentile(array, 98)
    plt.figure(figsize=(8, 8))
    im = plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im, label=title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plotly_time_series_chart(dates, values, title="NDVI Evolution"):
    """graphique interactif plotly — directement exploitable dans un dashboard dash."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode="lines+markers", name=title))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="NDVI", template="plotly_white")
    return fig


def dash_dashboard_skeleton(df: pd.DataFrame):
    """squelette minimal d'un dashboard dash pour restituer un suivi de parcelle."""
    from dash import Dash, dcc, html
    import plotly.express as px

    app = Dash(__name__)
    fig = px.line(df, x="date", y="ndvi", title="Suivi NDVI de la parcelle")
    app.layout = html.Div([
        html.H1("Dashboard de suivi agricole"),
        dcc.Graph(figure=fig),
    ])
    return app  # app.run(debug=True) pour lancer


# 8. pipeline / config / logging

import json
import logging


def setup_logger(name="agri_pipeline", level=logging.INFO):
    """
    logger standard pour tracer l'exécution d'un pipeline de traitement.
    le "if not logger.handlers" évite un bug classique : dupliquer chaque
    message si la fonction est appelée plusieurs fois dans la même session.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(json_path: str) -> dict:
    """charge une configuration externe (zone, dates, hyperparamètres) — évite le hardcoding."""
    with open(json_path, "r") as f:
        return json.load(f)


def save_config(config: dict, json_path: str):
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)


class ProcessingPipeline:
    """
    structure de pipeline minimaliste : une suite d'étapes nommées, exécutées
    dans l'ordre, avec logging à chaque étape (esprit orchestrateur type
    AWS Step Functions).

    exemple :
        pipeline = (
            ProcessingPipeline(name="ndvi_pipeline")
            .add_step("Load bands", load_bands_step)
            .add_step("Compute indices", compute_indices_step)
        )
        final_context = pipeline.run(initial_context)
    """
    def __init__(self, name="ndvi_pipeline"):
        self.logger = setup_logger(name)
        self.steps = []

    def add_step(self, step_name, func):
        self.steps.append((step_name, func))
        return self

    def run(self, context: dict) -> dict:
        for step_name, func in self.steps:
            self.logger.info(f"début étape : {step_name}")
            try:
                context = func(context)
                self.logger.info(f"fin étape : {step_name} — OK")
            except Exception as e:
                self.logger.error(f"échec étape {step_name} : {e}")
                raise
        return context


def upload_to_s3(local_path: str, bucket: str, s3_key: str):
    """upload un résultat (carte, modèle, csv) vers un bucket s3 — nécessite boto3 configuré."""
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"