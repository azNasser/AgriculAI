import toolkit as tk
import os
import numpy as np
import pandas as pd


# charger toutes les bandes

def load_bands_step(context):
    """charge chaque bande listée dans context["files"], et garde le profil
    géographique de la première comme référence pour la suite du pipeline."""
    folder = context["folder"]
    files = context["files"]
    bands = {}
    for name, file in files.items():
        data, profile = tk.load_band(os.path.join(folder, file))
        bands[name] = data
        if "reference_profile" not in context:
            context["reference_profile"] = profile
    context["bands"] = bands
    return context


# calculer tous les indices

def compute_indices_step(context):
    """calcule les 7 indices de végétation à partir des bandes chargées."""
    context["indices"] = tk.compute_all_indices(context["bands"])
    return context


# calculer les statistiques de chaque indice

def compute_stats_step(context):
    """résume chaque indice (moyenne, médiane, écart-type...) dans un dataframe,
    plus facile à lire et à exporter qu'une pile de dictionnaires séparés."""
    stats = []
    for name, array in context["indices"].items():
        s = tk.index_stats(array)
        s["index"] = name
        stats.append(s)
    context["stats"] = pd.DataFrame(stats)
    return context


# générer les cartes (images PNG)

def generate_maps_step(context):
    """exporte une carte PNG par indice dans le dossier de résultats."""
    output_dir = context.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    for name, array in context["indices"].items():
        path = os.path.join(output_dir, f"map_{name}.png")
        tk.plot_index_map(array, title=name, save_path=path)
    return context


# exporter le rapport final (csv + carte de synthèse)

def export_report_step(context):
    """exporte le tableau de statistiques en csv, et génère une carte de
    classification ndvi en 4 zones (eau / sol nu / végétation faible / dense)."""
    output_dir = context.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "index_statistics.csv")
    context["stats"].to_csv(csv_path, index=False)

    if "NDVI" in context["indices"]:
        # classify_ndvi_thresholds vient du toolkit, pour ne pas dupliquer
        # cette logique ici et dans app.py
        classes = tk.classify_ndvi_thresholds(context["indices"]["NDVI"])
        tk.plot_index_map(
            classes, title="NDVI classification", cmap="viridis",
            vmin=0, vmax=3,
            save_path=os.path.join(output_dir, "map_ndvi_classification.png")
        )

    print(f"rapport exporté dans : {output_dir}/")
    print(context["stats"])
    return context


# assemblage du pipeline

def build_pipeline():
    """assemble les 5 étapes dans l'ordre logique de traitement."""
    return (
        tk.ProcessingPipeline(name="vegetation_index_pipeline")
        .add_step("Load bands", load_bands_step)
        .add_step("Compute indices", compute_indices_step)
        .add_step("Compute statistics", compute_stats_step)
        .add_step("Generate maps", generate_maps_step)
        .add_step("Export report", export_report_step)
    )


# exécution en ligne de commande

if __name__ == "__main__":
    initial_context = {
        "folder": "images",
        "files": {
            "B02": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B02_(Raw).tiff",
            "B03": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B03_(Raw).tiff",
            "B04": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B04_(Raw).tiff",
            "B05": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B05_(Raw).tiff",
            "B08": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B08_(Raw).tiff",
            "B11": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B11_(Raw).tiff",
        },
        "output_dir": "results",
    }
    pipeline = build_pipeline()
    pipeline.run(initial_context)