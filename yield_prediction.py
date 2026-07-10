"""
module de démonstration : prédiction de rendement par machine learning.

utilise des données synthétiques générées de manière réaliste (pas de vrai
dataset externe à télécharger), pour que la démonstration reste autonome
dans un portfolio déployé publiquement. montre de bout en bout : feature
engineering temporel, split cohérent dans le temps, entraînement xgboost,
évaluation, importance des variables.
"""

import numpy as np
import pandas as pd
import toolkit as tk


# année de départ et nombre d'années générées par défaut. centralisés ici
# plutôt qu'en dur dans app.py, pour qu'il n'y ait qu'un seul endroit à
# changer si la période simulée doit évoluer un jour.
START_YEAR = 2018
DEFAULT_N_YEARS = 8


def get_available_years(n_years: int = DEFAULT_N_YEARS) -> list[int]:
    """
    renvoie la liste des années couvertes par le dataset synthétique, pour
    que app.py puisse construire son sélecteur d'année sans avoir à deviner
    ou dupliquer cette plage.

    exemple : get_available_years() -> [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    """
    return list(range(START_YEAR, START_YEAR + n_years))


def generate_synthetic_yield_dataset(n_years: int = DEFAULT_N_YEARS, n_fields: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    génère un jeu de données synthétique mais réaliste : pour chaque champ et
    chaque année, simule un ndvi de pic, une aire sous la courbe, un cumul de
    gdd et de pluie, puis calcule un rendement qui dépend de ces variables
    avec du bruit — pour que le modèle ait quelque chose de non trivial à apprendre.

    exemple : generate_synthetic_yield_dataset() -> dataframe de 320 lignes
    (8 années x 40 champs), avec les colonnes ndvi_peak, area_under_curve,
    gdd_cumulative, rain_cumulative, year, yield_t_ha
    """
    rng = np.random.default_rng(seed)
    rows = []

    for year in get_available_years(n_years):
        # une année plus ou moins favorable climatiquement, qui influence
        # tous les champs de cette année-là (corrélation inter-champs réaliste)
        year_quality = rng.normal(0, 1)

        for field_id in range(n_fields):
            ndvi_peak = np.clip(0.65 + 0.1 * year_quality + rng.normal(0, 0.08), 0.2, 0.95)
            area_under_curve = np.clip(45 + 8 * year_quality + rng.normal(0, 6), 15, 70)
            gdd_cumulative = np.clip(1400 + 80 * year_quality + rng.normal(0, 60), 1000, 1700)
            rain_cumulative = np.clip(320 + 40 * year_quality + rng.normal(0, 35), 150, 500)

            # rendement simulé comme une combinaison plausible des variables
            # ci-dessus, plus un bruit qui représente tout ce que le modèle
            # ne pourra jamais expliquer parfaitement (variabilité réelle)
            yield_t_ha = (
                2.0
                + 6.0 * ndvi_peak
                + 0.04 * area_under_curve
                + 0.001 * gdd_cumulative
                - 0.0008 * (rain_cumulative - 320) ** 2 / 100
                + rng.normal(0, 0.5)
            )
            yield_t_ha = round(max(yield_t_ha, 1.0), 2)

            rows.append({
                "year": year,
                "field_id": field_id,
                "ndvi_peak": round(ndvi_peak, 3),
                "area_under_curve": round(area_under_curve, 1),
                "gdd_cumulative": round(gdd_cumulative, 0),
                "rain_cumulative": round(rain_cumulative, 0),
                "yield_t_ha": yield_t_ha,
            })

    return pd.DataFrame(rows)


def run_yield_prediction_demo(test_year: int = None, 
                              n_estimators: int = 50, 
                              max_depth: int = 3, 
                              learning_rate: float = 0.1, 
                              subsample: float = 0.8) -> dict:
    """
    exécute la démonstration complète : génère les données, sépare
    entraînement/test par année (jamais aléatoirement, cf toolkit.temporal_split),
    entraîne un xgboost, et renvoie tout ce qu'il faut pour l'afficher dans l'app.

    si test_year n'est pas fourni, utilise automatiquement la dernière année
    disponible du dataset généré.

    exemple :
        result = run_yield_prediction_demo(test_year=2025)
        result["metrics"]             -> {"RMSE": 0.42, "MAE": 0.31, "R2": 0.87}
        result["feature_importance"]  -> dataframe trié
        result["predictions_df"]      -> comparaison prédiction vs réalité
    """
    df = generate_synthetic_yield_dataset()

    if test_year is None:
        test_year = get_available_years()[-1]

    if test_year not in df["year"].unique():
        # garde-fou explicite plutôt qu'un plantage silencieux plus loin
        # dans sklearn si le test set se retrouve vide
        raise ValueError(
            f"année {test_year} absente du dataset généré. "
            f"années disponibles : {get_available_years()}"
        )

    feature_cols = ["ndvi_peak", "area_under_curve", "gdd_cumulative", "rain_cumulative"]
    target_col = "yield_t_ha"

    train, test = tk.temporal_split(df, "year", test_years=[test_year])

    model, metrics, predictions = tk.train_xgboost_yield_model(
        train[feature_cols], train[target_col],
        test[feature_cols], test[target_col],
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample
    )

    importance = tk.feature_importance_table(model, feature_cols)

    # ... code existant ...
    predictions_df = test[["field_id", target_col]].copy()
    predictions_df["predicted_yield"] = predictions
    predictions_df["error"] = round(predictions_df["predicted_yield"] - predictions_df[target_col], 2)

    train_predictions = model.predict(train[feature_cols])
    train_predictions_df = train[["field_id", target_col]].copy()
    train_predictions_df["predicted_yield"] = train_predictions

    return {
        "dataset": df,
        "metrics": metrics,
        "feature_importance": importance,
        "predictions_df": predictions_df,
        "train_predictions_df": train_predictions_df,
        "test_year": test_year,
    }