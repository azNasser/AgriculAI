# Configuration de l'accès à l'API Copernicus (Sentinel-2)

L'onglet "Rechercher une nouvelle zone (ville)" de l'application récupère des
images Sentinel-2 en direct via l'API Copernicus (openEO). Pour que ça
fonctionne — en local comme une fois déployé — il faut créer une seule fois
un identifiant technique ("client OAuth"), gratuit.

Ce n'est **pas** un compte personnel classique : c'est un identifiant
machine-to-machine, qui ne demande à aucun visiteur de se connecter.

## Étape 1 — Créer un compte Copernicus Data Space (si pas déjà fait)

1. Va sur https://dataspace.copernicus.eu
2. Crée un compte gratuit

## Étape 2 — Créer un client OAuth (client_id / client_secret)

1. Va sur le tableau de bord Sentinel Hub, accessible depuis
   https://dataspace.copernicus.eu (section "Sentinel Hub" une fois connecté)
2. Cherche la section de gestion des utilisateurs / OAuth clients
   ("User settings" puis "OAuth clients", libellés selon les mises à jour du site)
3. Crée un nouveau client :
   - Nom : libre (ex : `agriculai-portfolio`)
   - Grant type : **Client Credentials** uniquement (décoche les autres options)
4. Copie le `client_id` et le `client_secret` générés — le secret ne sera
   affiché qu'une seule fois, sauvegarde-le immédiatement dans un endroit sûr

**Note** : ces identifiants sont documentés comme expirant après une certaine
durée (environ 90 jours d'après le support Copernicus) — il faudra en
régénérer un nouveau passé ce délai.

## Étape 3 — Configurer les identifiants en local

Crée un dossier `.streamlit/` à la racine du projet (s'il n'existe pas déjà),
puis un fichier `.streamlit/secrets.toml` avec ce contenu :

```toml
[copernicus]
client_id = "ton_client_id_ici"
client_secret = "ton_client_secret_ici"
```

**Ce fichier est volontairement exclu du `.gitignore` — ne le commite jamais.**

## Étape 4 — Configurer les identifiants sur Streamlit Cloud (déploiement)

1. Sur https://share.streamlit.io, ouvre les paramètres de ton app déployée
2. Va dans la section **"Secrets"**
3. Colle exactement le même contenu que ton `secrets.toml` local :

```toml
[copernicus]
client_id = "ton_client_id_ici"
client_secret = "ton_client_secret_ici"
```

4. Sauvegarde — l'app redémarre automatiquement avec les nouveaux secrets

## Vérification

Lance l'app en local :
```bash
streamlit run app.py
```
Choisis "Rechercher une nouvelle zone (ville)", entre un nom de ville, clique
sur le bouton. Si tout est bien configuré, une image récente s'affiche après
quelques secondes. En cas d'erreur explicite mentionnant les identifiants,
relis les étapes 2 et 3.
