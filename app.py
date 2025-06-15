import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import os
import gdown # Ajout pour le téléchargement depuis Google Drive

# --- ÉTAPE 1 : Chargement des données depuis Google Drive ---
# NOTE : Ce script va télécharger le fichier Parquet si il n'est pas trouvé localement.
GDRIVE_FILE_ID = "1znbv-o5XfyWLc_5IcnHosrDdgH8JafDp"
LOCAL_DATA_FILE = "data_sample_1million.parquet" # Nom du fichier qui sera sauvegardé localement

# On utilise un "décorateur" de Streamlit. @st.cache_data dit à l'application :
# "Exécute cette fonction une seule fois. Si on la rappelle avec les mêmes arguments,
# ne la recalcule pas, utilise directement le résultat que tu as mis en cache (en mémoire)."
# C'est très efficace pour ne pas re-télécharger et re-lire le fichier à chaque interaction de l'utilisateur.
@st.cache_data(ttl=24*3600)
def load_data():
    """
    Télécharge les données depuis Google Drive si elles ne sont pas disponibles localement,
    puis les charge dans un DataFrame Pandas.
    """
    # On vérifie d'abord si le fichier de données existe déjà sur le disque où l'application tourne.
    if not os.path.exists(LOCAL_DATA_FILE):
        try:
            # On construit l'URL de téléchargement direct à partir de l'ID du fichier Google Drive.
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            # On utilise la bibliothèque gdown pour télécharger le fichier.
            gdown.download(url, LOCAL_DATA_FILE, quiet=False)
            st.success("Téléchargement terminé.")
        except Exception as e:
            # Si le téléchargement échoue, on affiche un message d'erreur clair.
            st.error("ERREUR : Impossible de télécharger le fichier depuis Google Drive.")
            st.error(f"Vérifiez que le lien de partage est bien public ('Tous les utilisateurs disposant du lien').")
            st.error(f"Détails de l'erreur: {e}")
            # On retourne un DataFrame vide pour que l'application ne plante pas.
            return pd.DataFrame()
            
    try:
        # Une fois que le fichier est bien présent localement, on l'ouvre avec pandas.
        # Le format Parquet est très optimisé, c'est bien plus rapide à lire qu'un CSV.
        df = pd.read_parquet(LOCAL_DATA_FILE)
        # On s'assure que la colonne 'Cluster' est bien de type 'category'.
        # C'est une bonne pratique qui aide pandas et les bibliothèques de graphiques à mieux la gérer.
        df['Cluster'] = df['Cluster'].astype('category')
        return df
    except Exception as e:
        # Si la lecture du fichier échoue (fichier corrompu, etc.), on affiche une erreur.
        st.error(f"Erreur de lecture du fichier Parquet : {e}")
        return pd.DataFrame()

# --- Initialisation de l'application ---
# On configure la page pour qu'elle utilise toute la largeur de l'écran.
st.set_page_config(layout="wide")

# --- Introduction de l'application ---
# st.title et st.markdown sont les commandes de base pour écrire du texte dans Streamlit.
st.title("🚀 DUDA - Analyse de Cohortes v1")
st.markdown('''
### par Alexandre Cameron BORGES & Alioune DIOP

Nous avons conçu ce MVP d'application pour explorer des données clients et aider à prendre de meilleures décisions marketing. C'est une première version (MVP) que nous avons développée dans le cadre de notre formation au **DU Data Analytics de l'Université Panthéon Sorbonne**.

**Quel est l'objectif ?**
Dans un monde où le suivi des utilisateurs devient plus compliqué (fin des cookies tiers), il est essentiel de bien comprendre ses propres clients. Notre outil utilise les données internes d'une entreprise pour :
- **Identifier les produits et rayons qui marchent le mieux**.
- **Comprendre le comportement d'achat des clients** à travers des analyses comme le modèle RFM (Récence, Fréquence, Montant).
- **Segmenter la clientèle en groupes (clusters)** pour mieux cibler les actions marketing.

Pour cette démonstration, nous utilisons un jeu de données public d'Instacart provenant de Kaggle, qui contient 1 million de commandes.
''')

# On appelle notre fonction pour charger les données. Grâce au cache,
# cela ne sera réellement exécuté qu'une seule fois au démarrage.
df = load_data()
# Si le DataFrame est vide (à cause d'une erreur de chargement), on arrête l'exécution du script.
if df.empty:
    st.stop()

# --- Prétraitement & filtres dans la barre latérale ---
# st.sidebar permet de placer des éléments dans la barre latérale gauche.
st.sidebar.header("PARAMÈTRES D'ANALYSE")

jours = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
# On vérifie que la colonne 'order_dow' existe bien avant de l'utiliser.
if 'order_dow' in df.columns:
    # On crée une nouvelle colonne avec les noms des jours en toutes lettres pour que ce soit plus lisible.
    # On utilise pd.Categorical pour s'assurer que les jours restent dans le bon ordre.
    df['order_dow_name'] = pd.Categorical(df['order_dow'].map({i: jours[i] for i in range(7)}), categories=jours, ordered=True)
else:
    st.error("La colonne 'order_dow' est manquante dans les données.")
    st.stop()

# On crée les widgets interactifs dans la barre latérale.
# st.sidebar.multiselect permet de choisir plusieurs options dans une liste.
selected_days = st.sidebar.multiselect(
    "Jours de la semaine", options=jours, default=jours
)
# st.sidebar.slider crée un curseur pour sélectionner une plage de valeurs.
h_min, h_max = st.sidebar.slider(
    "Heure de la journée", int(df['order_hour_of_day'].min()), int(df['order_hour_of_day'].max()), (0, 23)
)
aisle_options = sorted(df['aisle'].unique().tolist())
cluster_options = sorted(df['Cluster'].unique().tolist())

selected_aisles = st.sidebar.multiselect(
    "Rayons (Aisles)", options=aisle_options, default=aisle_options
)
selected_clusters = st.sidebar.multiselect(
    "Clusters RFM", options=cluster_options, default=cluster_options
)

# Le cœur de l'interactivité est ici.
# On filtre le DataFrame principal en fonction des choix de l'utilisateur dans la sidebar.
# Le DataFrame 'filtered' ne contient que les lignes qui correspondent à tous les critères.
filtered = df[
    (df['order_dow_name'].isin(selected_days)) &
    (df['order_hour_of_day'].between(h_min, h_max)) &
    (df['aisle'].isin(selected_aisles)) &
    (df['Cluster'].isin(selected_clusters))
]

st.markdown("Nombre de commandes correspondant à nos filtres :")
# st.metric est un afficheur spécial pour les chiffres clés.
st.metric("Lignes sélectionnées", f"{len(filtered):,}")

# --- KPI globaux : prix moyen par produit, panier moyen, LTV moyenne ---
col1, col2, col3 = st.columns(3)

# 1) Prix moyen par produit 
avg_price_product = df['price'].mean()

# 2) Panier moyen par commande
order_totals_all = df.groupby('order_id')['price'].sum()
avg_order_value_all = order_totals_all.mean()

# 3) LTV moyenne par client (multiplication *11 pour être en phase avec le dataset complet incluant les commandes futures des clients)(en effet limite streamlit à 1M de lignes)
user_totals_all = df.groupby('user_id')['price'].sum()
avg_ltv_all = user_totals_all.mean()*11

# Affichage
col1.metric("Prix moyen global par produit", f"{avg_price_product:.2f} €")
col2.metric("Panier moyen global par commande (AOV)", f"{avg_order_value_all:.2f} €")
col3.metric("Dépense moyenne globale par client (LTV)", f"{avg_ltv_all:.2f} €")




# --- Section des Visualisations ---

st.header("🛒 Analyse des Ventes : Produits et Rayons")

# --- GRAPHIQUES PRODUITS AVEC PIE CHART ---
st.subheader("Quels sont nos produits vedettes ?")
st.markdown("Ici, on regarde les produits qui sont les plus vendus, à la fois en quantité et en chiffre d'affaires. Cela nous aide à identifier les produits stars.")
if not filtered.empty:
    # On calcule les top 12 produits en quantité vendue.
    top_qty_prod = filtered.groupby('product_name').size().nlargest(12)
    # On calcule les top 12 produits en chiffre d'affaires.
    top_rev_prod = filtered.groupby('product_name')['price'].sum().nlargest(12)
    # On fusionne les deux listes pour avoir une vision complète des produits importants.
    prods_union = pd.Index(top_qty_prod.index).union(top_rev_prod.index)
    # On recalcule les métriques (quantité et CA) uniquement pour cette liste de produits.
    df_prod = (
        filtered.groupby('product_name')
        .agg(total_qty=('order_id', 'count'), total_rev=('price', 'sum'))
        .loc[prods_union]
        .sort_values('total_qty', ascending=False)
    )
    # On crée une figure Matplotlib avec 2 graphiques (subplots) côte à côte.
    fig_prod, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [2, 1]})
    
    y_prod = list(range(len(df_prod)))
    # On dessine le premier graphique à barres (quantité).
    bars1 = ax1.barh([i - 0.2 for i in y_prod], df_prod['total_qty'], height=0.4, color='steelblue', label='Quantité vendue')
    ax1.set_yticks(y_prod); ax1.set_yticklabels(df_prod.index, fontsize=9); ax1.set_xlabel('Quantité vendue'); ax1.invert_yaxis()
    # ax1.twiny() crée un deuxième axe des abscisses (X) qui partage le même axe des ordonnées (Y).
    # C'est l'astuce pour afficher deux mesures différentes (quantité et CA) sur le même graphique.
    ax_tw = ax1.twiny()
    # On dessine le second graphique à barres (chiffre d'affaires).
    bars2 = ax_tw.barh([i + 0.2 for i in y_prod], df_prod['total_rev'], height=0.4, color='darkorange', label="Chiffre d'affaires (€)")
    ax_tw.set_xlabel("Chiffre d'affaires (€)")
    ax1.legend(handles=[bars1, bars2], labels=[h.get_label() for h in [bars1, bars2]], loc='lower right')
    ax1.set_title("Top produits : Quantité vs Chiffre d'affaires")

    # On dessine le second graphique : le diagramme circulaire (camembert).
    ordered_prod_rev = df_prod['total_rev'].sort_values(ascending=False)
    ax2.pie(
        ordered_prod_rev.values, autopct='%1.1f%%', labels=ordered_prod_rev.index,
        startangle=90, wedgeprops={'edgecolor': 'white'}
    )
    ax2.axis('equal'); ax2.set_title("Répartition du CA par produit\n(Top produits)")
    
    # On affiche la figure complète dans Streamlit.
    plt.tight_layout()
    st.pyplot(fig_prod)

    # --- MODIFICATION POUR LES BALLONS ---
    # On utilise st.session_state pour s'assurer que les ballons n'apparaissent qu'une seule fois par session.
    # st.session_state.get('balloons_shown') vérifie si la clé 'balloons_shown' existe.
    if not st.session_state.get('balloons_shown'):
        # Si elle n'existe pas, on lance les ballons.
        st.balloons()
        # Et on crée immédiatement la clé en la mettant à True pour ne pas que ça se reproduise.
        st.session_state.balloons_shown = True
    # --- FIN DE LA MODIFICATION ---

else:
    st.warning("Aucune donnée de produit à afficher avec les filtres actuels.")


# --- GRAPHIQUES RAYONS AVEC PIE CHART ---
st.subheader("Quels sont les rayons les plus populaires ?")
st.markdown("De la même manière que pour les produits, on analyse ici les rayons pour voir lesquels attirent le plus de clients et génèrent le plus de revenus.")
if not filtered.empty:
    # La logique est identique à celle des produits, mais en groupant par 'aisle' (rayon).
    top_qty_aisle = filtered.groupby('aisle').size().nlargest(12)
    top_rev_aisle = filtered.groupby('aisle')['price'].sum().nlargest(10)
    aisles_union = pd.Index(top_qty_aisle.index).union(top_rev_aisle.index)
    df_aisle = (
        filtered.groupby('aisle')
        .agg(total_qty=('order_id', 'count'), total_rev=('price', 'sum'))
        .loc[aisles_union]
        .sort_values('total_qty', ascending=False)
    )
    fig_aisle, (ax1_a, ax2_a) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})

    y_aisle = list(range(len(df_aisle)))
    bars1_a = ax1_a.barh([i - 0.2 for i in y_aisle], df_aisle['total_qty'], height=0.4, color='steelblue', label='Quantité vendue')
    ax1_a.set_yticks(y_aisle); ax1_a.set_yticklabels(df_aisle.index); ax1_a.set_xlabel('Quantité vendue'); ax1_a.invert_yaxis()
    ax2b_a = ax1_a.twiny()
    bars2_a = ax2b_a.barh([i + 0.2 for i in y_aisle], df_aisle['total_rev'], height=0.4, color='darkorange', label="Chiffre d'affaires (€)")
    ax2b_a.set_xlabel("Chiffre d'affaires (€)")
    ax1_a.legend(handles=[bars1_a, bars2_a], labels=[h.get_label() for h in [bars1_a, bars2_a]], loc='lower right')
    ax1_a.set_title("Top rayons : Quantité vs CA")

    ordered_aisle_rev = df_aisle['total_rev'].sort_values(ascending=False)
    ax2_a.pie(
        ordered_aisle_rev.values, labels=ordered_aisle_rev.index, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'white'}
    )
    ax2_a.axis('equal'); ax2_a.set_title("Répartition du CA par rayon\n(Top rayons)")
    
    plt.tight_layout()
    st.pyplot(fig_aisle)
else:
    st.warning("Aucune donnée de rayon à afficher avec les filtres actuels.")


# --- ANALYSE TEMPORELLE ---
st.header("🗓️ Analyse Temporelle : Quand les clients commandent-ils ?")
st.markdown("Cette section nous permet de visualiser les habitudes d'achat au fil de la semaine et de la journée, pour savoir quand nos clients sont les plus actifs.")
if not filtered.empty:
    # On calcule les données nécessaires : le nombre de commandes par heure et par jour.
    top_hours = filtered.groupby('order_hour_of_day').size()
    top_days = filtered.groupby('order_dow_name').size()
    # Pour la heatmap, on doit créer une "matrice" où les lignes sont les jours et les colonnes les heures.
    # .unstack() est la fonction pandas qui permet de faire cette transformation.
    sales_matrix = filtered.groupby(['order_dow_name', 'order_hour_of_day']).size().unstack(fill_value=0).reindex(jours)
    sales_matrix_k = sales_matrix / 1000 # On divise par 1000 pour un affichage plus lisible.

    # On crée une figure avec 3 graphiques.
    fig_temp, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1.5]})

    # Graphique 1 : Barres pour les jours.
    sns.barplot(x=top_days.index, y=top_days.values, palette='Greens', ax=axes[0])
    axes[0].set_title("Commandes par jour")
    axes[0].set_xlabel("Jour de la semaine")
    axes[0].set_ylabel("Nombre de produits commandés")
    axes[0].tick_params(axis='x', rotation=45)

    # Graphique 2 : Barres pour les heures.
    sns.barplot(x=top_hours.index, y=top_hours.values, palette='Greens', ax=axes[1])
    axes[1].set_title("Commandes par heure")
    axes[1].set_xlabel("Heure de la journée")
    axes[1].set_ylabel("Nombre de produits commandés")
    axes[1].tick_params(axis='x', rotation=45)

    # Graphique 3 : La heatmap (carte de chaleur).
    cmap_wg = LinearSegmentedColormap.from_list('RedGreen', ['#fdecec', '#2b8a3e'])
    sns.heatmap(
        sales_matrix_k, annot=True, fmt='.1f', cmap=cmap_wg, linewidths=0.5, linecolor='white',
        annot_kws={'fontsize': 8}, cbar_kws={'label': 'Commandes (en milliers)'}, ax=axes[2]
    )
    axes[2].set_title("Fréquentation par jour et heure")
    axes[2].set_xlabel("Heure de la journée")
    axes[2].set_ylabel("Jour de la semaine")
    
    plt.tight_layout()
    st.pyplot(fig_temp)
else:
    st.warning("Aucune donnée pour l'analyse temporelle avec les filtres actuels.")


st.header("👥 Analyse des Clients par Clusters (Modèle RFM)")
st.markdown("Le modèle RFM (Récence, Fréquence, Montant) est une technique marketing puissante pour comprendre le comportement des clients. Nous l'utilisons pour classer les clients en différents groupes (clusters), comme les 'champions', les 'clients à risque', etc. Cela nous permet d'adapter nos stratégies de communication pour chaque groupe.")

# --- PAIRPLOT ---
st.subheader("Vue d'ensemble des relations entre les indicateurs RFM")
st.info("ℹ️ Pour que l'affichage reste rapide, ce graphique est calculé sur un échantillon aléatoire des données sélectionnées.")
if not filtered.empty:
    # On prend un petit échantillon des données (5000 lignes) car ce graphique est très lent à calculer.
    SAMPLE_SIZE_PAIRPLOT = 5000
    # .reset_index(drop=True) est une sécurité pour éviter des bugs d'index avec pandas.
    df_for_pairplot = filtered.sample(n=min(len(filtered), SAMPLE_SIZE_PAIRPLOT), random_state=42).reset_index(drop=True)
    
    # sns.pairplot crée une matrice de graphiques pour visualiser les relations entre chaque paire de variables.
    g = sns.pairplot(
        df_for_pairplot, vars=["Recency", "Frequency", "Monetary"], hue="Cluster", palette="Set2"
    )
    g.fig.suptitle("Pairplot des variables RFM par Cluster", y=1.03)
    st.pyplot(g)
else:
    st.warning("Aucune donnée à afficher pour le pairplot.")


# --- GRAPHIQUES 3D ---
st.subheader("Exploration 3D des segments de clients")
# On utilise st.columns pour afficher les deux graphiques 3D côte à côte.
col3d_1, col3d_2 = st.columns(2)

with col3d_1:
    st.markdown("##### Vue d'ensemble statique")
    st.info("ℹ️ Cette vue est une image fixe, calculée sur un échantillon pour plus de rapidité.")
    if not filtered.empty:
        # On prend un échantillon de 10 000 lignes pour ce graphique.
        SAMPLE_SIZE_STATIC_3D = 10000
        df_for_plot_static = filtered.sample(n=min(len(filtered), SAMPLE_SIZE_STATIC_3D), random_state=42).reset_index(drop=True)
        
        # On crée un dictionnaire qui associe chaque numéro de cluster à une couleur.
        unique_clusters = sorted(df_for_plot_static['Cluster'].unique())
        palette_static = sns.color_palette("Set2", len(unique_clusters))
        color_dict = dict(zip(unique_clusters, palette_static))
        
        fig_static = plt.figure(figsize=(8, 6))
        ax_static = fig_static.add_subplot(111, projection='3d')
        
        # On dessine les points pour chaque cluster un par un, dans une boucle.
        # C'est une méthode plus robuste qui évite certains bugs de pandas.
        for cluster, color in color_dict.items():
            cluster_data = df_for_plot_static[df_for_plot_static['Cluster'] == cluster]
            ax_static.scatter(
                cluster_data["Recency"], 
                cluster_data["Frequency"], 
                cluster_data["Monetary"],
                c=[color],
                s=20, 
                alpha=0.6,
                label=f'Cluster {cluster}' # Le 'label' est important pour créer la légende automatiquement.
            )

        ax_static.set_xlabel("Recency")
        ax_static.set_ylabel("Frequency")
        ax_static.set_zlabel("Monetary")
        ax_static.legend(title="Cluster", fontsize='small')
        st.pyplot(fig_static)

with col3d_2:
    st.markdown("##### Vue interactive pour l'exploration")
    st.info("ℹ️ Vous pouvez tourner, zoomer et survoler les points pour voir les détails. Ce graphique utilise aussi un échantillon.")
    if not filtered.empty:
        # On peut se permettre un échantillon plus grand (50 000) car Plotly est très optimisé pour le web.
        SAMPLE_SIZE_INTERACTIVE_3D = 50000
        df_for_plot_interactive = filtered.sample(n=min(len(filtered), SAMPLE_SIZE_INTERACTIVE_3D), random_state=42).reset_index(drop=True)

        # Pour être sûr que les couleurs sont les mêmes que sur les autres graphiques,
        # on crée une "map" de couleurs explicite.
        all_clusters = sorted(df['Cluster'].unique())
        hex_colors = sns.color_palette("Set2", len(all_clusters)).as_hex()
        color_discrete_map = {cluster: color for cluster, color in zip(all_clusters, hex_colors)}
        
        # On crée le graphique 3D avec Plotly Express, c'est très direct.
        fig_interactive = px.scatter_3d(
            df_for_plot_interactive, 
            x='Recency', y='Frequency', z='Monetary', 
            color='Cluster',
            color_discrete_map=color_discrete_map, # On passe notre map de couleurs ici.
            title=f"Clusters RFM ({len(df_for_plot_interactive):,} points)"
        )
        
        # Quelques ajustements pour rendre le graphique plus joli.
        fig_interactive.update_traces(marker=dict(size=3, opacity=0.7))
        fig_interactive.update_layout(margin=dict(l=0, r=0, b=0, t=40), legend=dict(orientation="h", yanchor="bottom", y=0.01))
        st.plotly_chart(fig_interactive, use_container_width=True)

# --- AJOUT DU TABLEAU RECAPITULATIF ---
st.header("💡 Synthèse globale (full_dataset) par Segment")
st.markdown("Ce tableau résume les caractéristiques de chaque segment de clientèle et propose des pistes d'actions concrètes pour chacun.")

st.markdown("""
| Cluster | Nom du Segment | Récence (Moyenne) | Fréquence (Moyenne) | Montant (Moyen) | Nb de Clients | Interprétations | Priorités |
|:---:|:---|:---:|:---:|:---:|:---:|:---|:---|
| 2 | **🌱 Nouveaux Clients / Prometteurs** | 4 | 6 | 177.90€ | 104,382 | Ont acheté très récemment mais peu souvent. Potentiel de croissance. | **C'est votre futur.** Mettez en place des processus solides pour transformer ces nouveaux acheteurs en clients réguliers et, à terme, en champions. |
| 0 | **👤 Clients Occasionnels** | 10 | 15 | 483.90€ | 58,169 | Achètent de temps en temps, sans grande fréquence ni dépense. | **Maintenez le contact** via des actions automatisées et à faible coût. Ils constituent une base stable qui peut réagir aux offres de masse. |
| 3 | **⚠️ Clients à Risque / Sur le départ** | 15 | 32 | 1,162.30€ | 33,628 | Clients de valeur qui n'ont pas acheté depuis longtemps. Risque de churn. | **Agissez MAINTENANT** pour retenir ces clients de valeur. C'est souvent plus rentable de retenir un client que d'en acquérir un nouveau. |
| 1 | **🏆 Champions / Meilleurs Clients** | 13 | 64 | 2,829.40€ | 10,030 | Très fidèles et dépensent beaucoup. Le cœur de votre chiffre d'affaires. | **Chouchoutez ce groupe.** Ils financent la croissance. Assurez-vous qu'ils restent heureux et fidèles. |
""")
# --- FIN DE L'AJOUT ---

st.header("Analyse de Texte : Nuage de Mots")
import re
from wordcloud import WordCloud, STOPWORDS

# 1. Charger le texte de l’article
text = """
RFM Segmentation: What Is It, and How Can Marketers Use It?
RFM Segmentation: What Is It, and How Can Marketers Use It? 
By Ethan Shust, Last Updated: June 2, 2023

Audience segmentation is a powerful way to analyze your customer data, define your target customer persona(s), and tailor your acquisition/retention efforts to each unique persona for the best-performing results. RFM Segmentation is a method of segmenting your customer audiences by grouping them based on three key vectors: recency, frequency, and monetary value. Let’s dig in!

The Foundation: Marketing Segmentation 101
Before we dive into the depths of RFM Segmentation (Recency, Frequency, Monetary Value), let's take a step back and review what segmentation means for marketers.

Segmentation is a crucial component of every team's marketing strategy. It involves dividing a larger market into smaller groups of consumers with similar characteristics. By doing this, businesses can identify different customer groups and develop targeted marketing strategies that cater to each group's specific wants and needs. When done correctly, this leads to higher engagement, conversion rates, and customer loyalty.

Marketers often use the following methods to segment their customers:

Demographic Segmentation: This method groups customers based on demographic factors, such as age, gender, income, education, occupation, and family status.
Geographic Segmentation: This method groups customers based on their location, such as country, region, city, or neighborhood.
Psychographic Segmentation: This method groups customers based on personality traits, values, beliefs, interests, and lifestyles.
Behavioral Segmentation: This method groups customers based on their behaviors, such as buying habits, product usage, brand loyalty, and decision-making processes.
For more information on each type of segmentation, check out this awesome article from Yieldify.


From the above list, behavioral segmentation is the method that informs RFM audiences, as these audiences are developed based on customers' past purchase behavior.

But before we dive into RFM audiences, here’s a short list of the ways that analyzing transactional data benefits your business:

Clear, Objective Criteria: Transactional data uncovers the clear, objective criteria critical for identifying and targeting high-value customers.
Accurate and Reliable Data: Transactional data is typically more accurate and reliable than other forms of data, such as survey data or demographic data. Other forms of data can be subject to self-reporting biases and general errors.
More Actionable Insights: Transactional data provides more actionable insights into customer behavior and preferences, allowing businesses to develop targeted and effective marketing strategies.
Immediate Feedback: Transactional data provides immediate feedback on the effectiveness of marketing campaigns and promotions. This allows businesses to make quick optimizations necessary for improving results.
So now that you’re sold on why transactional data matters, let’s talk RFM.


The RFM model
RFM segmentation is a customer segmentation technique that categorizes customers based on three vectors of their past purchasing behaviors: Recency, Frequency, and Monetary Value. We’ve unpacked each of these vectors for you below.

Recency: This refers to how much time has elapsed since a customer last made a purchase. In most cases, customers who have purchased more recently are valuable customers likely to engage with the brand again.
Frequency: This refers to how often a customer makes purchases. Customers who make frequent purchases are typically a brand’s most valuable customers, demonstrating higher brand loyalty than those who purchase less often.
Monetary Value: This refers to the amount of money a customer has spent on purchases over a given period of time. Of course, customers who spend a comparatively high amount of money are valuable customers with higher brand loyalty, and are most likely to purchase again. Customers with high monetary value are typically the most valuable customers for a given brand.
Using the RFM segmentation method, customers are ranked based on each of these three vectors and then divided - or segmented - into groups based on their scores.

For example, a customer who purchased within the past few days, purchases your product at least once a month, and spends a comparatively high amount of money with your business will be segmented into a high-value segment.

Conversely, a customer who hasn’t purchased from you in a year with only two low-value historical purchases would be segmented into a low-value segment.

‍

Common RFM Segments
Here are some of the most popular segments brands develop using the RFM model:

High-Value Customers: These are customers who score high on all three metrics (Recency, Frequency, and Monetary Value). They are typically the most engaged and loyal customers, likely to make frequent, high-value purchases. Businesses can target these customers with exclusive offers, loyalty programs, and personalized marketing campaigns to keep them engaged and encourage them to continue buying from the brand in the future.
New Customers: These are customers who have made a recent purchase (high score for Recency), but have not yet established a high level of loyalty or spending (low scores for Frequency and Monetary Value). Businesses can target these customers with welcome offers and promotions to encourage them to make another purchase and/or establish a stronger personal relationship with the brand.
High Risk Customers: These are customers who have not made a purchase recently (low score for Recency), but have previously made frequent and high-value purchases (high scores for Frequency and Monetary Value). These customers are segmented to flag them as they likely need some sort of nurturing to get them to return to the brand. Businesses can use this segment to launch re-engagement campaigns, personalized offers, and loyalty programs to encourage future purchases.
Low-Value Customers: These are customers who score low on all three metrics (Recency, Frequency, and Monetary Value). They are typically the least engaged and loyal customers and are unlikely to make frequent or high-value purchases. Businesses can target these customers with promotions and special offers to encourage them to make another purchase, or they can be excluded from marketing campaigns to focus on higher-value customers.
‍

Building an RFM analysis
Now that you know the what and the why, it’s time to dig into the how-to.

PS: Triple Whale offers pre-built, plug & play RFM segments for your analysis if you’re ready to get rocking and rolling. If you’re feeling more manual, check out the steps below.

Gather transactional data from your customer database or CRM system. This data should include purchase histories such as date, frequency, and transaction value for each customer.
Calculate RFM scores by assigning a score for each of the three metrics (Recency, Frequency, and Monetary Value) based on the customer's past purchase behavior. For instance, a recent purchase earns a high Recency score, while frequent purchases result in a high frequency score.
After calculating RFM scores for each customer, you can segment customers into groups based on their scores. One common approach is to use quartiles to divide customers into four equal groups based on their scores for each metric.
After segmenting customers, analyzing their characteristics and behavior can reveal patterns and trends. This can assist in improving marketing and customer engagement strategies.
Develop marketing strategies for each customer segment. High-value customers can be targeted with exclusive offers and loyalty programs, while low-value customers can be targeted with promotions to encourage repeat purchases.
Regularly monitor and refine your marketing strategies and RFM segmentation based on customer behavior to improve customer engagement and maximize customer value.
This is admittedly a very time-consuming process…so if you’re looking for a faster path to insights, keep reading to learn how Triple Whale unlocks ready-to-go RFM Analysis.

‍

Triple Whale’s SCDP and RFM audiences
For Shopify-based brands, Triple Whale’s Smart Customer Data Platform comes pre-loaded with 6 powerful RFM segments. These are Triple Whale’s own AI-generated RFM audiences.

To build these audiences, Triple Whale examines all of your customers’ historical data and splits them into buckets based on RFM scoring. In Triple Whale’s SCDP, these segments are highlighted with a square AI icon.

Triple Whale’s RFM audiences are defined as follows:

Loyal = Customers who buy the most often from your store.
Core = Highly engaged customers who have bought most recently, the most often, and have generated the most revenue.
Newbies = First-time buyers on your site.
Whales = Customers who have generated the most revenue for your store.
Promising = Customers who return often, but do not spend a lot.
Lost = Customers who have made one purchase but have not been known to return.
Daniel Okon, Founder/CEO of ACTIV Agency, recently tweeted about these RFM audiences in action:


Daniel is the Founder/CEO of ACTIV, an agency that helps 7 and 8-figure brands unlock the key components that help them scale profitably. He also has a strong Twitter game. Check him out!

‍

The RFM Model for Marketers
For marketers, leveraging the RFM Model unlocks detailed insights into customer behavior and preferences based directly on transactional data. With these insights, brands can assess customer engagement and loyalty levels, identify high loss-risk customers, and strategically tailor their marketing strategies to each unique persona. Additionally, RFM Analysis provides businesses with the data necessary to determine which channels, campaigns, and marketing tactics result in the highest return on investment (ROI).

In today’s crazy, competitive market, RFM segmentation & analysis is an extremely powerful tool for achieving every brand’s main goal: making more money to grow the business.

Ready to start digging in on RFM? Check out Triple Whale’s Smart Customer Data Platform and start exploring your brand’s smart RFM audiences today.

Book A Free Demo Today!


"""

# afficher le texte original dans une section dépliable
with st.expander("Cliquez pour voir le texte original"):
    st.write(text)

# 2. Nettoyage du texte
#   - On supprime tous les caractères non alphabétiques (ponctuation, chiffres, etc.)
#   - On convertit en minuscules pour uniformiser
text_clean = re.sub(r'[^A-Za-z\s]', '', text).lower()

# 3. Tokenisation
#   - On découpe le texte en mots (tokens) sur les espaces
tokens = text_clean.split()

# 4. Suppression des stopwords
#   - Les stopwords sont des mots très fréquents (articles, prépositions…) 
#     qui n’apportent pas de sens particulier à l’analyse
stopwords = set(STOPWORDS)
tokens_filtered = [t for t in tokens if t not in stopwords]

# 5. Génération du nuage de mots
processed_text = " ".join(tokens_filtered)
wc = WordCloud(width=800, height=400).generate(processed_text)

# 6. Affichage (avec Streamlit)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15, 7.5))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption("Application réalisée par Alexandre Cameron BORGES & Alioune DIOP pour le DU Panthéon Sorbonne Data Analytics.")
