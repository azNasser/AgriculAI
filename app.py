import streamlit as st
import toolkit as tk
import acquisition as acq
import yield_prediction as yp
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Suivi végétation Sentinel-2", layout="wide")

st.title("Suivi de la végétation par télédétection Sentinel-2")
st.markdown("Calcul interactif d'indices de végétation à partir de bandes satellite.")


@st.cache_data
def load_sample_data():
    """charge les 6 bandes d'exemple fournies avec le dépôt, sans appel réseau."""
    folder = "images"
    files = {
        "B02": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B02_(Raw).tiff",
        "B03": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B03_(Raw).tiff",
        "B04": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B04_(Raw).tiff",
        "B05": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B05_(Raw).tiff",
        "B08": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B08_(Raw).tiff",
        "B11": "2026-07-08-00_00_2026-07-08-23_59_Sentinel-2_L2A_B11_(Raw).tiff",
    }
    bands = {}
    for name, file in files.items():
        data, _ = tk.load_band(os.path.join(folder, file))
        bands[name] = data
    return bands


def apply_scl_mask(bands: dict, scl: np.ndarray) -> dict:
    """filtre les bandes selon la scl (nuages, ombres...) quand elle est disponible."""
    mask = ~np.isin(scl, tk.SCL_CLASSES_TO_EXCLUDE)
    return {name: np.where(mask, array, np.nan) for name, array in bands.items()}


def kmeans_zoning(indices: dict, n_clusters: int = 3) -> np.ndarray:
    """
    regroupe les pixels en zones homogènes via un véritable algorithme de
    machine learning (k-means non supervisé), à partir de plusieurs indices
    combinés (ndvi, ndwi, gndvi) plutôt qu'un seuil fixe choisi à la main.
    l'algorithme découvre lui-même les regroupements naturels dans les données.

    les clusters sont ensuite triés par ndvi moyen croissant, pour que le
    label 0 corresponde toujours à la zone la moins vigoureuse.
    """
    feature_names = ["NDVI", "NDWI", "GNDVI"]
    stack = np.stack([indices[name] for name in feature_names], axis=-1)
    shape = stack.shape[:2]
    flat = stack.reshape(-1, len(feature_names))
    valid = ~np.isnan(flat).any(axis=1)

    labels_flat = np.full(flat.shape[0], np.nan)
    if valid.sum() >= n_clusters:
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        raw_labels = model.fit_predict(flat[valid])

        # trie les clusters par ndvi moyen croissant, pour un résultat lisible
        # (label 0 = zone la plus faible, label max = zone la plus vigoureuse)
        ndvi_means = [flat[valid][raw_labels == k, 0].mean() for k in range(n_clusters)]
        order = np.argsort(ndvi_means)
        remap = {old: new for new, old in enumerate(order)}
        labels_flat[valid] = [remap[label] for label in raw_labels]

    return labels_flat.reshape(shape)


INDEX_INFO = {
    "NDVI": {
        "formula": "(PIR - Rouge) / (PIR + Rouge)",
        "meaning": "mesure la vigueur générale de la végétation, via le contraste "
                   "entre l'absorption du rouge par la chlorophylle et la réflexion "
                   "du proche infrarouge par la structure des feuilles.",
        "range": "0.6 à 0.9 pour une végétation dense et saine, 0 à 0.2 pour du sol nu, négatif pour l'eau.",
    },
    "NDRE": {
        "formula": "(PIR - RedEdge) / (PIR + RedEdge)",
        "meaning": "même principe que le ndvi, mais reste discriminant sur une "
                   "végétation déjà très dense, là où le ndvi sature et n'augmente plus.",
        "range": "0.3 à 0.5 sur une culture dense, alors que le ndvi y serait déjà proche de son maximum.",
    },
    "GNDVI": {
        "formula": "(PIR - Vert) / (PIR + Vert)",
        "meaning": "plus sensible à la concentration en chlorophylle sur la durée, "
                   "utile pour suivre un statut azoté (l'azote est un composant "
                   "essentiel de la chlorophylle).",
        "range": "0.3 à 0.5 en cas de carence azotée probable, 0.6 à 0.8 sur une culture bien fertilisée.",
    },
    "NDWI": {
        "formula": "(PIR - SWIR) / (PIR + SWIR)",
        "meaning": "ne mesure pas la vigueur mais la teneur en eau de la plante. "
                   "Peut baisser avant même que le ndvi ne bouge, signe précoce de stress hydrique.",
        "range": "0.3 à 0.5 sur une végétation bien hydratée, 0.1 à 0.25 en cas de stress hydrique.",
    },
    "SAVI": {
        "formula": "((PIR - Rouge) / (PIR + Rouge + 0.5)) x 1.5",
        "meaning": "comme le ndvi, mais corrige la contribution du sol nu visible "
                   "entre de jeunes plants — utile en début de cycle (levée, tallage).",
        "range": "0.15 à 0.3 sur une culture jeune et clairsemée, 0.4 à 0.6 sur une végétation dense.",
    },
    "EVI": {
        "formula": "2.5 x (PIR - Rouge) / (PIR + 6xRouge - 7.5xBleu + 1)",
        "meaning": "comme le ndvi, corrige en plus des perturbations atmosphériques "
                   "(via le bleu) et résiste mieux à la saturation.",
        "range": "0.5 à 0.8 sur une végétation très dense, après filtrage des valeurs aberrantes.",
    },
    "MSAVI2": {
        "formula": "(2xPIR+1 - racine((2xPIR+1)^2 - 8x(PIR-Rouge))) / 2",
        "meaning": "variante du savi qui calcule elle-même son propre facteur de "
                   "correction du sol, sans avoir à le régler à la main.",
        "range": "proche du savi, un peu plus stable sur des sols de réflectance variable.",
    },
}


def show_index_glossary():
    """affiche une explication de chaque indice (formule, sens, plage de valeurs)
    dans un panneau repliable, pour ne pas alourdir l'interface par défaut."""
    with st.expander("Que mesure chaque indice ? (cliquer pour déplier)"):
        for name, info in INDEX_INFO.items():
            st.markdown(
                f"**{name}** — `{info['formula']}`  \n"
                f"{info['meaning']}  \n"
                f"*Plage typique : {info['range']}*"
            )
            st.markdown("---")


def show_kmeans_explanation():
    """explique le principe du clustering k-means appliqué ici, pour qu'un
    visiteur du portfolio comprenne ce que fait le modèle sans avoir à lire le code."""
    with st.expander("Comment fonctionne le clustering K-Means ici ? (cliquer pour déplier)"):
        st.markdown(
            "Chaque pixel de la parcelle est décrit par 3 nombres : son NDVI, "
            "son NDWI et son GNDVI. K-Means place ainsi chaque pixel comme un "
            "point dans un espace à 3 dimensions, puis regroupe les points qui "
            "se ressemblent le plus en N groupes (le nombre choisi avec le curseur).\n\n"
            "Contrairement aux seuils manuels (qui ne regardent que le NDVI et "
            "des bornes fixées à la main), le clustering peut séparer deux "
            "pixels ayant un NDVI proche mais un NDWI très différent — par "
            "exemple une zone en début de stress hydrique, même si sa vigueur "
            "générale semble encore normale.\n\n"
            "Les groupes obtenus sont ensuite triés par NDVI moyen croissant, "
            "pour que la couleur la plus sombre corresponde toujours à la "
            "zone la moins vigoureuse, quel que soit le nombre de clusters choisi."
        )


def display_yield_prediction_tab():
    """
    onglet de démonstration d'un vrai modèle de machine learning supervisé
    (xgboost), séparé du reste de l'app car il ne dépend d'aucune image
    satellite : utilise des données synthétiques mais réalistes, générées
    à la volée, pour rester utilisable sans configuration supplémentaire.
    """
    st.markdown(
        "Démonstration d'un modèle de prédiction de rendement entraîné sur "
        "des données synthétiques (ndvi de pic, aire sous la courbe, cumul "
        "de gdd, cumul de pluie). La séparation entraînement/test se fait "
        "par année, jamais aléatoirement, afin d'éviter toute fuite "
        "d'information du futur vers le passé."
    )

    available_years = yp.get_available_years()
    test_year = st.selectbox("Année à utiliser comme test", available_years, index=len(available_years) - 1)

    if st.button("Entraîner le modèle"):
        with st.spinner("Génération des données et entraînement du modèle..."):
            result = yp.run_yield_prediction_demo(test_year=test_year)
        st.session_state["yield_result"] = result

    if "yield_result" in st.session_state:
        result = st.session_state["yield_result"]

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE (t/ha)", result["metrics"]["RMSE"])
        col2.metric("MAE (t/ha)", result["metrics"]["MAE"])
        col3.metric("R2", result["metrics"]["R2"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Importance des variables")
            importance_df = result["feature_importance"]
            fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
            ax_imp.barh(importance_df["feature"], importance_df["importance"])
            ax_imp.invert_yaxis()  # la feature la plus importante en haut
            ax_imp.set_xlabel("Importance")
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        with col_b:
            st.subheader(f"Prédiction vs réalité ({result['test_year']})")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(result["predictions_df"]["yield_t_ha"], result["predictions_df"]["predicted_yield"])
            lims = [
                min(result["predictions_df"]["yield_t_ha"].min(), result["predictions_df"]["predicted_yield"].min()),
                max(result["predictions_df"]["yield_t_ha"].max(), result["predictions_df"]["predicted_yield"].max()),
            ]
            ax.plot(lims, lims, linestyle="--", color="gray")
            ax.set_xlabel("Rendement réel (t/ha)")
            ax.set_ylabel("Rendement prédit (t/ha)")
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Voir le détail des prédictions par champ"):
            # st.table dépend aussi de pyarrow en interne. si l'erreur DLL
            # revient, remplacer la ligne suivante par :
            # st.text(result["predictions_df"].to_string(index=False))
            st.table(result["predictions_df"])


def display_results(bands: dict, meta: dict | None = None):
    """
    bloc d'affichage réutilisé pour l'exemple statique et pour une recherche
    par ville. ne relance jamais de calcul réseau : uniquement des calculs
    locaux sur des données déjà en mémoire, donc peut être rappelé librement
    à chaque interaction (changement de curseur, de menu...).
    """
    indices = tk.compute_all_indices(bands)

    if meta:
        st.success(f"Image du {meta['date']} — {meta['city']} (lat {meta['lat']}, lon {meta['lon']})")

    st.sidebar.markdown("---")
    selected_index = st.sidebar.selectbox("Indice à afficher", list(indices.keys()), key="selected_index")
    show_index_glossary()

    # curseur à double poignée : les deux bornes ne peuvent jamais s'inverser,
    # contrairement à deux curseurs séparés
    st.sidebar.subheader("Seuils de classification NDVI")
    soil_threshold, dense_threshold = st.sidebar.slider(
        "Sol nu <-> végétation faible <-> végétation dense",
        min_value=0.0, max_value=1.0, value=(0.2, 0.5), step=0.05,
        key="ndvi_thresholds",
        help="La zone entre les deux poignées correspond à la végétation faible."
    )
    st.sidebar.caption(
        f"sol nu : ndvi < {soil_threshold}  \n"
        f"végétation faible : {soil_threshold} à {dense_threshold}  \n"
        f"végétation dense : ndvi > {dense_threshold}"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Carte — {selected_index}")
        array = indices[selected_index]
        fig, ax = plt.subplots(figsize=(7, 7))
        # échelle de couleur adaptée à la vraie distribution de cette image,
        # plutôt qu'une échelle fixe -1/1 qui écraserait le contraste (ex: ndwi)
        vmin = np.nanpercentile(array, 2)
        vmax = np.nanpercentile(array, 98)
        im = ax.imshow(array, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Statistiques")
        stats = tk.index_stats(array)
        for key, value in stats.items():
            st.metric(key, value)

    st.subheader("Zonage de la parcelle")
    zoning_method = st.radio(
        "Méthode de zonage",
        ["Seuils manuels", "Clustering automatique (K-Means, machine learning)"],
        horizontal=True,
    )
    if zoning_method == "Clustering automatique (K-Means, machine learning)":
        show_kmeans_explanation()

    ndvi_array = indices["NDVI"]

    if zoning_method == "Seuils manuels":
        # classify_ndvi_thresholds vient du toolkit, pour ne pas dupliquer
        # cette logique ici et dans pipeline.py
        classes = tk.classify_ndvi_thresholds(ndvi_array, soil_threshold, dense_threshold)
        n_classes = 4
        caption = "Zones définies à la main selon les deux seuils NDVI ci-dessus."
    else:
        n_clusters = st.slider("Nombre de zones (clusters)", 2, 6, 3, 1)
        classes = kmeans_zoning(indices, n_clusters=n_clusters)
        n_classes = n_clusters
        caption = (
            "Zones découvertes automatiquement par K-Means à partir du NDVI, "
            "NDWI et GNDVI combinés — aucun seuil fixé à la main ici."
        )

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.imshow(classes, cmap="viridis", vmin=0, vmax=n_classes - 1)
    ax2.axis("off")
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(caption)

    valid_pixels = np.sum(~np.isnan(classes))
    if valid_pixels > 0 and zoning_method == "Seuils manuels":
        dense_pct = round(100 * np.sum(classes == 3) / valid_pixels, 1)
        st.info(f"{dense_pct}% de la zone est classée en végétation dense avec ces seuils.")


tab_satellite, tab_yield = st.tabs(["Suivi satellite", "Prédiction de rendement (machine learning)"])

with tab_satellite:
    st.sidebar.header("Source des données")
    source = st.sidebar.radio(
        "Choisir la source",
        ["Zone d'exemple (incluse dans le dépôt)", "Rechercher une nouvelle zone (ville)"],
    )

    if source == "Zone d'exemple (incluse dans le dépôt)":
        # pas besoin de session_state ici : load_sample_data() est déjà mis
        # en cache par st.cache_data, donc rappeler cette fonction ne
        # recharge jamais les fichiers depuis le disque après le premier appel
        bands = load_sample_data()
        display_results(bands)

    else:
        st.subheader("Rechercher une zone par ville")
        # le formulaire permet de valider avec la touche entrée, pas seulement au clic
        with st.form("city_search_form"):
            city_name = st.text_input("Nom de la ville", placeholder="Ex : Chartres, Toulouse, Orléans...")
            submitted = st.form_submit_button("Récupérer la dernière image disponible")

        if submitted and city_name:
            with st.spinner(f"Recherche de la dernière image satellite la moins nuageuse sur {city_name}..."):
                try:
                    bands, scl, meta = acq.fetch_latest_image(city_name)
                    if scl is not None:
                        bands = apply_scl_mask(bands, scl)
                    # stocké en mémoire de session : survit aux changements de
                    # curseur/menu, qui ne redéclenchent pas ce bloc "if submitted"
                    st.session_state["city_bands"] = bands
                    st.session_state["city_meta"] = meta
                except RuntimeError as e:
                    st.error(str(e))
                    st.info("Voir SETUP.md pour configurer l'accès à l'API Copernicus.")
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Erreur lors de la récupération de l'image : {e}")

        elif submitted and not city_name:
            st.warning("Entre un nom de ville avant de lancer la recherche.")

        # affiché à chaque exécution du script (donc aussi lors des
        # changements de curseur/menu), tant qu'une recherche a déjà réussi
        # au moins une fois
        if "city_bands" in st.session_state:
            display_results(st.session_state["city_bands"], st.session_state["city_meta"])

with tab_yield:
    display_yield_prediction_tab()