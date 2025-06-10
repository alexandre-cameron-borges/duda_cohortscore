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
GDRIVE_FILE_ID = "104HyALPq0dAvM41KE0KswKIcIDukgXZb"
LOCAL_DATA_FILE = "instacart_sample_500k.parquet" # Nom du fichier qui sera sauvegardé localement

@st.cache_data(ttl=24*3600)
def load_data():
    """
    Télécharge les données depuis Google Drive si elles ne sont pas disponibles localement,
    puis les charge dans un DataFrame Pandas.
    """
    if not os.path.exists(LOCAL_DATA_FILE):
        st.info(f"Fichier de données non trouvé. Téléchargement depuis Google Drive...")
        try:
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, LOCAL_DATA_FILE, quiet=False)
            st.success("Téléchargement terminé.")
        except Exception as e:
            st.error("ERREUR : Impossible de télécharger le fichier depuis Google Drive.")
            st.error(f"Vérifiez que le lien de partage est bien public ('Tous les utilisateurs disposant du lien').")
            st.error(f"Détails de l'erreur: {e}")
            return pd.DataFrame()
            
    try:
        df = pd.read_parquet(LOCAL_DATA_FILE)
        # S'assurer que le cluster est une catégorie pour les couleurs et les légendes
        df['Cluster'] = df['Cluster'].astype('category')
        return df
    except Exception as e:
        st.error(f"Erreur de lecture du fichier Parquet : {e}")
        return pd.DataFrame()

# --- Initialisation de l'application ---
st.set_page_config(layout="wide")
st.title("Analyse Interactive de la Cohorte Instacart")

df = load_data()

if df.empty:
    st.stop()

# --- Prétraitement & filtres dans la barre latérale ---
st.sidebar.header("Filtres de l'Analyse")

jours = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
# S'assurer que la colonne existe avant de l'utiliser
if 'order_dow' in df.columns:
    df['order_dow_name'] = pd.Categorical(df['order_dow'].map({i: jours[i] for i in range(7)}), categories=jours, ordered=True)
else:
    st.error("La colonne 'order_dow' est manquante dans les données.")
    st.stop()

selected_days = st.sidebar.multiselect(
    "Jours de la semaine", options=jours, default=jours
)
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
# Application des filtres
filtered = df[
    (df['order_dow_name'].isin(selected_days)) &
    df['order_hour_of_day'].between(h_min, h_max) &
    (df['aisle'].isin(selected_aisles)) &
    (df['Cluster'].isin(selected_clusters))
]

st.metric("Lignes sélectionnées", f"{len(filtered):,}")

# --- Section des Visualisations ---

st.header("Analyse des Produits et Rayons")

# --- GRAPHIQUES PRODUITS AVEC PIE CHART ---
st.subheader("Analyse des Top Produits")
if not filtered.empty:
    top_qty_prod = filtered.groupby('product_name').size().nlargest(12)
    top_rev_prod = filtered.groupby('product_name')['price'].sum().nlargest(10)
    prods_union = pd.Index(top_qty_prod.index).union(top_rev_prod.index)
    df_prod = (
        filtered.groupby('product_name')
        .agg(total_qty=('order_id', 'count'), total_rev=('price', 'sum'))
        .loc[prods_union]
        .sort_values('total_qty', ascending=True)
    )
    # Création de la figure avec deux panneaux
    fig_prod, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [2, 1]})
    
    # Panel 1: Barres horizontales
    y_prod = list(range(len(df_prod)))
    bars1 = ax1.barh([i - 0.2 for i in y_prod], df_prod['total_qty'], height=0.4, color='steelblue', label='Quantité vendue')
    ax1.set_yticks(y_prod); ax1.set_yticklabels(df_prod.index, fontsize=9); ax1.set_xlabel('Quantité vendue'); ax1.invert_yaxis()
    ax_tw = ax1.twiny()
    bars2 = ax_tw.barh([i + 0.2 for i in y_prod], df_prod['total_rev'], height=0.4, color='darkorange', label="Chiffre d'affaires (€)")
    ax_tw.set_xlabel("Chiffre d'affaires (€)")
    ax1.legend(handles=[bars1, bars2], labels=[h.get_label() for h in [bars1, bars2]], loc='lower right')
    ax1.set_title("Top produits : Quantité vs Chiffre d'affaires")

    # Panel 2: Pie chart
    ordered_prod_rev = df_prod['total_rev'].sort_values(ascending=False)
    ax2.pie(
        ordered_prod_rev.values, autopct='%1.1f%%', labels=ordered_prod_rev.index,
        startangle=90, wedgeprops={'edgecolor': 'white'}
    )
    ax2.axis('equal'); ax2.set_title("Répartition du CA par produit\n(Top produits)")
    
    plt.tight_layout()
    st.pyplot(fig_prod)
else:
    st.warning("Aucune donnée de produit à afficher avec les filtres actuels.")


# --- GRAPHIQUES RAYONS AVEC PIE CHART ---
st.subheader("Analyse des Top Rayons")
if not filtered.empty:
    top_qty_aisle = filtered.groupby('aisle').size().nlargest(12)
    top_rev_aisle = filtered.groupby('aisle')['price'].sum().nlargest(10)
    aisles_union = pd.Index(top_qty_aisle.index).union(top_rev_aisle.index)
    df_aisle = (
        filtered.groupby('aisle')
        .agg(total_qty=('order_id', 'count'), total_rev=('price', 'sum'))
        .loc[aisles_union]
        .sort_values('total_qty', ascending=True)
    )
    # Création de la figure avec deux panneaux
    fig_aisle, (ax1_a, ax2_a) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})

    # Panel 1: Barres horizontales
    y_aisle = list(range(len(df_aisle)))
    bars1_a = ax1_a.barh([i - 0.2 for i in y_aisle], df_aisle['total_qty'], height=0.4, color='steelblue', label='Quantité vendue')
    ax1_a.set_yticks(y_aisle); ax1_a.set_yticklabels(df_aisle.index); ax1_a.set_xlabel('Quantité vendue'); ax1_a.invert_yaxis()
    ax2b_a = ax1_a.twiny()
    bars2_a = ax2b_a.barh([i + 0.2 for i in y_aisle], df_aisle['total_rev'], height=0.4, color='darkorange', label="Chiffre d'affaires (€)")
    ax2b_a.set_xlabel("Chiffre d'affaires (€)")
    ax1_a.legend(handles=[bars1_a, bars2_a], labels=[h.get_label() for h in [bars1_a, bars2_a]], loc='lower right')
    ax1_a.set_title("Top rayons : Quantité vs CA")

    # Panel 2: Pie chart
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


# --- MODIFICATION: Restauration des histogrammes jour/heure ---
st.header("Analyse Temporelle des Commandes")
if not filtered.empty:
    # Calcul des données pour les graphiques
    top_hours = filtered.groupby('order_hour_of_day').size()
    top_days = filtered.groupby('order_dow_name').size()
    sales_matrix = filtered.groupby(['order_dow_name', 'order_hour_of_day']).size().unstack(fill_value=0).reindex(jours)
    sales_matrix_k = sales_matrix / 1000

    # Création de la figure avec 3 sous-graphes
    fig_temp, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1.5]})

    # 1) Histogramme : produits commandés par jour
    sns.barplot(x=top_days.index, y=top_days.values, palette='Greens', ax=axes[0])
    axes[0].set_title("Produits commandés par jour")
    axes[0].set_xlabel("Jour de la semaine")
    axes[0].set_ylabel("Nombre de produits")
    axes[0].tick_params(axis='x', rotation=45)

    # 2) Histogramme : produits commandés par heure
    sns.barplot(x=top_hours.index, y=top_hours.values, palette='Greens', ax=axes[1])
    axes[1].set_title("Produits commandés par heure")
    axes[1].set_xlabel("Heure de la journée")
    axes[1].set_ylabel("Nombre de produits")
    axes[1].tick_params(axis='x', rotation=45)

    # 3) Heatmap : commandes jour x heure en milliers
    cmap_wg = LinearSegmentedColormap.from_list('RedGreen', ['#fdecec', '#2b8a3e'])
    sns.heatmap(
        sales_matrix_k, annot=True, fmt='.1f', cmap=cmap_wg, linewidths=0.5, linecolor='white',
        annot_kws={'fontsize': 8}, cbar_kws={'label': 'Commandes (en milliers)'}, ax=axes[2]
    )
    axes[2].set_title("Commandes par jour et heure (en milliers)")
    axes[2].set_xlabel("Heure de la journée")
    axes[2].set_ylabel("Jour de la semaine")
    
    plt.tight_layout()
    st.pyplot(fig_temp)
else:
    st.warning("Aucune donnée pour l'analyse temporelle avec les filtres actuels.")


st.header("Analyse Détaillée des Clusters RFM")

# --- PAIRPLOT ---
st.subheader("Relations entre les variables RFM (Pairplot)")
st.info("ℹ️ Ce graphique est généré sur un échantillon pour garantir la fluidité.")
if not filtered.empty:
    SAMPLE_SIZE_PAIRPLOT = 5000
    # FIX: Ajout de .reset_index() pour éviter les erreurs d'index
    df_for_pairplot = filtered.sample(n=min(len(filtered), SAMPLE_SIZE_PAIRPLOT), random_state=42).reset_index(drop=True)
    
    g = sns.pairplot(
        df_for_pairplot, vars=["Recency", "Frequency", "Monetary"], hue="Cluster", palette="Set2"
    )
    g.fig.suptitle("Pairplot des variables RFM par Cluster", y=1.03)
    st.pyplot(g)
else:
    st.warning("Aucune donnée à afficher pour le pairplot.")


# --- GRAPHIQUES 3D ---
st.subheader("Visualisation 3D des Clusters")
col3d_1, col3d_2 = st.columns(2)

with col3d_1:
    st.markdown("##### Vue 3D Statique (Matplotlib)")
    st.info("ℹ️ Échantillon de 10 000 points max.")
    if not filtered.empty:
        SAMPLE_SIZE_STATIC_3D = 10000
        df_for_plot_static = filtered.sample(n=min(len(filtered), SAMPLE_SIZE_STATIC_3D), random_state=42).reset_index(drop=True)
        
        unique_clusters = sorted(df_for_plot_static['Cluster'].unique())
        palette_static = sns.color_palette("Set2", len(unique_clusters))
        color_dict = dict(zip(unique_clusters, palette_static))
        
        fig_static = plt.figure(figsize=(8, 6))
        ax_static = fig_static.add_subplot(111, projection='3d')
        
        for cluster, color in color_dict.items():
            cluster_data = df_for_plot_static[df_for_plot_static['Cluster'] == cluster]
            ax_static.scatter(
                cluster_data["Recency"], 
                cluster_data["Frequency"], 
                cluster_data["Monetary"],
                c=[color],
                s=20, 
                alpha=0.6,
                label=f'Cluster {cluster}'
            )

        ax_static.set_xlabel("Recency")
        ax_static.set_ylabel("Frequency")
        ax_static.set_zlabel("Monetary")
        ax_static.legend(title="Cluster", fontsize='small')
        st.pyplot(fig_static)

with col3d_2:
    st.markdown("##### Vue 3D Interactive (Plotly)")
    st.info("ℹ️ Échantillon de 50 000 points max.")
    if not filtered.empty:
        SAMPLE_SIZE_INTERACTIVE_3D = 50000
        df_for_plot_interactive = filtered.sample(n=min(len(filtered), SAMPLE_SIZE_INTERACTIVE_3D), random_state=42).reset_index(drop=True)

        # --- MODIFICATION POUR LA COHÉRENCE DES COULEURS ---
        # 1. Créer un dictionnaire de couleurs unique basé sur tous les clusters possibles
        all_clusters = sorted(df['Cluster'].unique())
        hex_colors = sns.color_palette("Set2", len(all_clusters)).as_hex()
        color_discrete_map = {cluster: color for cluster, color in zip(all_clusters, hex_colors)}
        
        # 2. Passer ce dictionnaire à Plotly
        fig_interactive = px.scatter_3d(
            df_for_plot_interactive, 
            x='Recency', y='Frequency', z='Monetary', 
            color='Cluster',
            color_discrete_map=color_discrete_map, # Utiliser le dictionnaire de couleurs explicite
            title=f"Clusters RFM ({len(df_for_plot_interactive):,} points)"
        )
        # --- FIN DE LA MODIFICATION ---
        
        fig_interactive.update_traces(marker=dict(size=3, opacity=0.7))
        fig_interactive.update_layout(margin=dict(l=0, r=0, b=0, t=40), legend=dict(orientation="h", yanchor="bottom", y=0.01))
        st.plotly_chart(fig_interactive, use_container_width=True)

# --- WORDCLOUD OPTIONNEL ---
st.sidebar.markdown("---")
st.sidebar.subheader("Analyse de Texte (Optionnel)")
article_file = st.sidebar.file_uploader("Uploader un fichier .txt", type=["txt"])
if article_file:
    st.header("WordCloud du texte fourni")
    try:
        text = article_file.read().decode('utf-8')
        wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(12,6))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    except Exception as e:
        st.error(f"Erreur lors de la génération du WordCloud : {e}")

# --- Footer ---
st.markdown("---")
st.caption("Application Streamlit ACB & Alioune.")
