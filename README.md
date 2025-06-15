# DUDA CohortScore v1

> Le DUDA CohortScore est le MVP d’une Webapp Streamlit d’analyse de cohortes, réalisée pour le DU Sorbonne Data Analytics 2024-2025 par Alexandre Cameron BORGES & Alioune DIOP.
> L’outil segmente la clientèle Instacart via l’analyse RFM (Récence, Fréquence, Montant), explore les habitudes d’achat et génère un WordCloud à partir d’un texte importé.

<p align="center">
  <a href="https://github.com/alexandre-cameron-borges/duda_cohortscore" target="_blank">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repo-181717?logo=github">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python">
</p>

---

## · 1️⃣ ✨ Objectif

* **Segmenter** les clients selon le modèle RFM.
* **Visualiser** les tendances d’achat (top produits, heatmaps temporelles).
* **Explorer** dynamiquement via filtres (jour, heure, rayon, cluster).
* **Générer** un WordCloud à partir d’un fichier texte lié au e-commerce.

## · 2️⃣ 🚀 Démo rapide

Ouvrez la WebApp hébergée → https://acb-dudacohortscore.streamlit.app/

ou

1. Cloner le dépôt et installer les dépendances (cf. §6).
2. Lancer l’application :

   ```bash
   streamlit run app.py
   ```
3. Dans l’interface :

   * Sélectionner le fichier `data/instacart_sample_1m.parquet`
   * Appliquer des filtres (jour, heure, rayon, cluster)
   * Explorer les graphiques (bar charts, heatmap, pairplot, vue 3D)
   * Charger un texte pour générer un WordCloud

## · 3️⃣ 📊 Jeux de données

| Table                                    |       Lignes | Description                       |
| ---------------------------------------- | -----------: | --------------------------------- |
| [`instacart_sample_1m.parquet`](https://drive.google.com/file/d/1znbv-o5XfyWLc_5IcnHosrDdgH8JafDp/view?usp=drive_link)            |    1 000 000 | Ce jeu de données provient d'Instacart, une plateforme américaine de livraison d'épicerie en ligne. Il contient des informations anonymisées sur plus de 3 millions de commandes passées par plus de 200 000 utilisateurs Instacart en 2017.  |

**Contenu principal
**Le dataset comprend plusieurs fichiers CSV interconnectés :

Orders : Informations sur chaque commande (ID utilisateur, jour de la semaine, heure, délai depuis la dernière commande)
Products : Catalogue de ~50 000 produits avec leurs noms et rayons
Aisles : Les 134 rayons du magasin (ex: "fresh vegetables", "packaged cheese")
Departments : Les 21 départements (ex: "dairy eggs", "beverages")
Order_products : Détails des produits dans chaque commande avec l'ordre d'ajout au panier
    

*Source* : [Kaggle – Instacart Online Grocery Basket Analysis](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)

## · 4️⃣ 🧠 Méthodologie

1. **EDA & nettoyage** (`data_and_finetuning_cohort.py`) :

   * Détection et traitement des NaN, doublons, typage.
2. **Création de variables** :

   * `order_dow_name` (jour de la semaine), totaux par client, indicateurs RFM.
3. **Clustering RFM** :

   * Standardisation, méthode du coude, KMeans.
4. **Visualisations** (10 graphiques) :

   * Bar charts, heatmap, pairplot, vue 3D.
5. **Text mining** :

   * Nettoyage, tokenisation, suppression de stopwords, génération de WordCloud.

## · 5️⃣ 🏗️ Architecture de l’application

```
duda_cohortscore/
├── app.py                          # Application Streamlit
├── data_and_finetuning_cohort.py  # EDA, nettoyage & création de variables
├── requirements.txt                # Dépendances Python
├── data/
│   ├── [instacart_sample_1m.parquet](https://drive.google.com/file/d/1znbv-o5XfyWLc_5IcnHosrDdgH8JafDp/view?usp=drive_link) # Échantillon de travail
│   └── [instacart_cleaned.csv](https://drive.google.com/file/d/1pRjgJ3X8CXfcaApv_W-QfZiWTHO71HlS/view?usp=drive_link)       # Jeu nettoyé final
└── README.md                       # Ce document
```

## · 6️⃣ ⚙️ Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/alexandre-cameron-borges/duda_cohortscore.git
cd duda_cohortscore

# 2. Créer et activer l’environnement
python -m venv .venv && source .venv/bin/activate  # macOS/Linux
# venv\Scripts\activate                            # Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer l’application
streamlit run app.py
```

> **Note** : l’application s’attend à trouver `data/instacart_sample_1m.parquet`. Ajustez le chemin dans `app.py` si nécessaire.

## · 7️⃣ 🙋 Auteurs

* **Alexandre Cameron BORGES** – [LinkedIn](https://fr.linkedin.com/in/alexandre-cameron-borges)
* **Alioune DIOP**

> *“In God We Trust, All the Others We Want Data.”*

*Dernière mise à jour : 10 juin 2025*
