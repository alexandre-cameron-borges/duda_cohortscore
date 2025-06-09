pip install -r requirements.txt 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Chargement des données depuis Kaggle ---
@st.cache(allow_output_mutation=True)
def load_data():
    output_csv = 'instacart_cleaned.csv'
    # Si le CSV n'existe pas localement, le télécharger depuis Kaggle
    if not os.path.exists(output_csv):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'alexandrecameronb/duda-cohort',
            path='.',
            unzip=True
        )
    df = pd.read_csv(output_csv)
    return df

# Chargement des données
df = load_data()

# --- Sidebar : filtres dynamiques ---
st.sidebar.header("Filtres")
jours = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
df['order_dow_name'] = df['order_dow'].map({i: jours[i] for i in range(7)})
selected_days = st.sidebar.multiselect(
    "Jours de la semaine", options=df['order_dow_name'].unique(), default=df['order_dow_name'].unique()
)
h_min, h_max = st.sidebar.slider(
    "Heure de la journée", int(df['order_hour_of_day'].min()), int(df['order_hour_of_day'].max()), (0, 23)
)
selected_aisles = st.sidebar.multiselect(
    "Rayons", options=df['aisle'].unique(), default=df['aisle'].unique()
)
selected_clusters = st.sidebar.multiselect(
    "Segments RFM", options=df['Cluster'].unique().tolist(), default=df['Cluster'].unique().tolist()
)

# Application des filtres
filtered = df[
    (df['order_dow_name'].isin(selected_days)) &
    (df['order_hour_of_day'] >= h_min) & (df['order_hour_of_day'] <= h_max) &
    (df['aisle'].isin(selected_aisles)) &
    (df['Cluster'].isin(selected_clusters))
]

# --- Titre de l'app ---
st.title("Exploration Interactive du Dataset Instacart")

# 1) Top 10 Produits par Quantité vendue
st.subheader("Top 10 Produits par Quantité")
fig1, ax1 = plt.subplots()
top_qty = filtered['product_name'].value_counts().nlargest(10)
ax1.barh(top_qty.index, top_qty.values)
ax1.invert_yaxis()
ax1.set_xlabel("Quantité vendue")
st.pyplot(fig1)

# 2) Top 10 Produits par Chiffre d'affaires
st.subheader("Top 10 Produits par CA")
fig2, ax2 = plt.subplots()
top_rev = filtered.groupby('product_name')['price'].sum().nlargest(10)
ax2.barh(top_rev.index, top_rev.values)
ax2.invert_yaxis()
ax2.set_xlabel("Chiffre d'affaires (€)")
st.pyplot(fig2)

# 3) Analyse par Rayon
st.subheader("Quantité vs CA par Rayon")
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
df_aisle = filtered.groupby('aisle').agg(qty=('order_id', 'count'), rev=('price', 'sum')).nlargest(10, 'qty')
axes[0].barh(df_aisle.index, df_aisle['qty']); axes[0].set_title("Quantité"); axes[0].invert_yaxis()
axes[1].barh(df_aisle.index, df_aisle['rev']); axes[1].set_title("CA (€)"); axes[1].invert_yaxis()
st.pyplot(fig3)

# 4) Heatmap Jour x Heure
st.subheader("Heatmap : Commandes par Jour et Heure (en milliers)")
sales_matrix = filtered.groupby(['order_dow_name', 'order_hour_of_day']).size().unstack(fill_value=0)
sales_matrix_k = sales_matrix / 1000
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.heatmap(sales_matrix_k, annot=False, cmap='Greens', ax=ax4)
ax4.set_xlabel("Heure"); ax4.set_ylabel("Jour")
st.pyplot(fig4)

# 5) Clusters RFM interactif
st.subheader("Clusters RFM (Scatter 3D interactif)")
fig5 = px.scatter_3d(
    filtered,
    x='Recency', y='Frequency', z='Monetary',
    color='Cluster', size_max=6, opacity=0.7,
    title="RFM 3D"
)
st.plotly_chart(fig5)

# 6) WordCloud du texte d'article
st.subheader("WordCloud du Texte")
with open('article.txt', 'r', encoding='utf-8') as f:
    text = f.read()
w = WordCloud(width=800, height=400, background_color='white').generate(text)
st.image(w.to_array(), use_column_width=True)

# Footer
st.markdown("---")
st.caption("App construite avec Streamlit pour explorer le dataset Instacart")


