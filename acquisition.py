"""
Acquisition automatisée de bandes Sentinel-2 via l'API Copernicus (openEO),
à partir d'un simple nom de ville.

Authentification par "client credentials" (identifiants machine-to-machine) :
aucune connexion requise pour les visiteurs de l'app — seul le propriétaire
du projet configure une fois un identifiant technique (voir SETUP.md).
"""

import os
import requests
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import openeo


REQUIRED_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11", "SCL"]


def geocode_city(city_name: str):
    """
    Convertit un nom de ville en coordonnées (lat, lon) via l'API gratuite
    Nominatim (OpenStreetMap), sans clé d'API nécessaire.

    Exemple : geocode_city("Toulouse") -> (43.6047, 1.4442, "Toulouse, France")
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "portfolio-agri-app (contact via github)"}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"Ville introuvable : {city_name}")
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    full_name = results[0].get("display_name", city_name)
    return lat, lon, full_name


def build_bbox(lat: float, lon: float, size_km: float = 3.0):
    """
    Construit une bbox (en degrés) d'environ size_km x size_km autour du point.
    La conversion km -> degrés de longitude dépend de la latitude (les
    méridiens se rapprochent en s'éloignant de l'équateur), d'où le cos(lat).
    """
    delta_lat = (size_km / 2) / 111.0
    delta_lon = (size_km / 2) / (111.0 * np.cos(np.radians(lat)))
    return {
        "west": lon - delta_lon,
        "east": lon + delta_lon,
        "south": lat - delta_lat,
        "north": lat + delta_lat,
    }


@st.cache_resource(show_spinner=False)
def _openeo_connection():
    """
    Connexion authentifiée par identifiants client (pas d'interaction
    utilisateur). Nécessite les secrets [copernicus] client_id / client_secret
    configurés (voir SETUP.md) — sinon lève une erreur explicite.
    """
    if "copernicus" not in st.secrets:
        raise RuntimeError(
            "Identifiants Copernicus non configurés. "
            "Voir SETUP.md pour créer un client OAuth et le déclarer dans les secrets Streamlit."
        )
    connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc_client_credentials(
        client_id=st.secrets["copernicus"]["client_id"],
        client_secret=st.secrets["copernicus"]["client_secret"],
    )
    return connection


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_latest_image(city_name: str, search_days: int = 45, max_cloud_cover: int = 30, size_km: float = 3.0):
    """
    Récupère la bande la plus récente disponible avec peu de nuages sur la
    zone correspondant au nom de ville donné.

    Renvoie :
        bands (dict[str, np.ndarray]) : réflectance (0-1) par bande, hors SCL
        scl (np.ndarray | None)       : la bande de classification qualité, si disponible
        meta (dict)                   : informations sur la recherche (ville, date, bbox...)
    """
    lat, lon, full_name = geocode_city(city_name)
    bbox = build_bbox(lat, lon, size_km)

    end_date = pd.Timestamp.utcnow().date()
    start_date = end_date - pd.Timedelta(days=search_days)

    connection = _openeo_connection()

    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[str(start_date), str(end_date)],
        bands=REQUIRED_BANDS,
        max_cloud_cover=max_cloud_cover,
    )

    cache_file = f"_cache_{city_name.replace(' ', '_')}.nc"
    datacube.download(cache_file)

    ds = xr.open_dataset(cache_file)

    if ds.sizes.get("t", 0) == 0:
        ds.close()
        os.remove(cache_file)
        raise ValueError(
            f"Aucune image disponible pour {full_name} sur les {search_days} derniers jours "
            f"avec moins de {max_cloud_cover}% de nuages. Essaie une autre ville ou élargis la période."
        )

    latest_date = ds.t.values[-1]
    ds_latest = ds.sel(t=latest_date)

    bands = {}
    for band_name in REQUIRED_BANDS:
        if band_name == "SCL":
            continue
        bands[band_name] = ds_latest[band_name].values.astype(float) / 10000

    scl = ds_latest["SCL"].values if "SCL" in ds_latest else None

    meta = {
        "city": full_name,
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "date": str(latest_date)[:10],
        "bbox": bbox,
    }

    # Le fichier local n'est qu'un intermediaire de telechargement : on l'a
    # deja charge en memoire ci-dessus, donc on le supprime tout de suite
    # plutot que de laisser s'accumuler des .nc sur le disque du serveur
    # a chaque nouvelle ville recherchee.
    ds.close()
    os.remove(cache_file)

    return bands, scl, meta