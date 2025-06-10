import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Upload manuel du CSV ---
st.title("Exploration Interactive du Dataset Instacart")
uploaded_csv = st.sidebar.file_uploader(
    "Télécharger le CSV nettoyé", type=["csv"]
)
if uploaded_csv is None:
    st.warning("Veuillez uploader le fichier 'instacart_cleaned.csv' pour continuer.")
    st.stop()
# Lecture du CSV uploadé
try:
    df = pd.read_csv(uploaded_csv)
except Exception as e:
    st.error(f"Erreur lors de la lecture du CSV : {e}")
    st.stop()

# --- Exploitation du DataFrame ---
# Ajout du nom de jour
jours = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
df['order_dow_name'] = df['order_dow'].map({i: jours[i] for i in range(7)})

# Paramètres de filtre
st.sidebar.header("Filtres")
selected_days = st.sidebar.multiselect(
    "Jours de la semaine", options=df['order_dow_name'].unique(), default=list(df['order_dow_name'].unique())
)
h_min, h_max = st.sidebar.slider(
    "Heure de la journée", int(df['order_hour_of_day'].min()), int(df['order_hour_of_day'].max()), (0, 23)
)
selected_aisles = st.sidebar.multiselect(
    "Rayons", options=df['aisle'].unique(), default=list(df['aisle'].unique())
)
selected_clusters = st.sidebar.multiselect(
    "Segments RFM", options=df['Cluster'].unique().tolist(), default=df['Cluster'].unique().tolist()
)

# Filtrage
filtered = df[
    (df['order_dow_name'].isin(selected_days)) &
    df['order_hour_of_day'].between(h_min, h_max) &
    (df['aisle'].isin(selected_aisles)) &
    (df['Cluster'].isin(selected_clusters))
]

# --- Visualisations ---
# 1) Top 10 Produits par Quantité
st.subheader("Top 10 Produits par Quantité")
fig1, ax1 = plt.subplots()
top_qty = filtered['product_name'].value_counts().nlargest(10)
ax1.barh(top_qty.index, top_qty.values)
ax1.invert_yaxis()
ax1.set_xlabel("Quantité vendue")
st.pyplot(fig1)

# 2) Top 10 Produits par CA
st.subheader("Top 10 Produits par Chiffre d'affaires")
fig2, ax2 = plt.subplots()
top_rev = filtered.groupby('product_name')['price'].sum().nlargest(10)
ax2.barh(top_rev.index, top_rev.values)
ax2.invert_yaxis()
ax2.set_xlabel("Chiffre d'affaires (€)")
st.pyplot(fig2)

# 3) Quantité vs CA par Rayon
st.subheader("Quantité vs CA par Rayon")
fig3, (ax_qty, ax_rev) = plt.subplots(1, 2, figsize=(12, 5))
df_aisle = filtered.groupby('aisle').agg(qty=('order_id', 'count'), rev=('price', 'sum')).nlargest(10, 'qty')
ax_qty.barh(df_aisle.index, df_aisle['qty'])
ax_qty.invert_yaxis(); ax_qty.set_title("Quantité")
ax_rev.barh(df_aisle.index, df_aisle['rev'])
ax_rev.invert_yaxis(); ax_rev.set_title("CA (€)")
st.pyplot(fig3)

# 4) Heatmap Jour x Heure
st.subheader("Heatmap : Commandes par Jour et Heure (en milliers)")
sales_matrix = filtered.groupby(['order_dow_name','order_hour_of_day']).size().unstack(fill_value=0)
sales_matrix_k = sales_matrix / 1000
fig4, ax4 = plt.subplots(figsize=(8,4))
sns.heatmap(sales_matrix_k, annot=False, cmap='Greens', ax=ax4)
ax4.set_xlabel("Heure"); ax4.set_ylabel("Jour")
st.pyplot(fig4)

# 5) Scatter 3D interactif Plotly des clusters RFM
st.subheader("Clusters RFM (Scatter 3D interactif)")
fig5 = px.scatter_3d(
    filtered, x='Recency', y='Frequency', z='Monetary',
    color='Cluster', size_max=6, opacity=0.7, title="RFM 3D"
)
st.plotly_chart(fig5, use_container_width=True)

# 6) WordCloud du texte
st.subheader("WordCloud du Texte (optionnel)")
article_file = st.sidebar.file_uploader(
    "Uploader l'article texte (facultatif)", type=["txt"]
)
if article_file:
    try:
        text = article_file.read().decode('utf-8')
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wc.to_array(), use_column_width=True)
    except Exception as e:
        st.error(f"Erreur WordCloud : {e}")

# Footer
st.markdown("---")
st.caption("App Streamlit avec visualisations et upload manuel de données")
st.markdown("---")
st.caption("App Streamlit avec visualisations et upload manuel de données")
