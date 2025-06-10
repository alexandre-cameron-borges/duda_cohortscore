import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# --- Chargement des données depuis Kaggle ---
@st.cache_data(ttl=24*3600)
def load_data():
    output_csv = 'instacart_cleaned.csv'
    if not os.path.exists(output_csv):
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'alexandrecameronb/duda-cohort',
                path='.',
                unzip=True
            )
        except Exception as e:
            st.error(
                "Impossible de télécharger le CSV depuis Kaggle.\n"
                "Vérifiez vos secrets KAGGLE_USERNAME/KAGGLE_KEY,\n"
                "ou placez 'instacart_cleaned.csv' manuellement dans le répertoire.\n"
                f"Erreur: {e}"
            )
            return pd.DataFrame()
    return pd.read_csv(output_csv)

# Chargement des données
df = load_data()
if df.empty:
    st.error(
        "Aucune donnée n'a pu être chargée.\n"
        "Vérifiez Kaggle API ou la présence du CSV local."
    )
    st.stop()

# --- Visualisations statiques du notebook ---
st.header("Analyses Exploratoires")

# 1) Top produits : dual-axe Quantité vs CA et camembert
st.subheader("Top produits : Quantité vs Chiffre d'affaires")
# Calcul
prod_qty = df.groupby('product_name')['order_id'].count().rename('total_qty')
prod_rev = df.groupby('product_name')['price'].sum().rename('total_rev')
top_products = pd.concat([prod_qty, prod_rev], axis=1).nlargest(12, 'total_qty')
# Figure
fig_prod, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios':[2,1]})
y_pos = range(len(top_products))
# bar dual-axe
df_plot = top_products.sort_values('total_qty')
bars1 = ax1.barh([i-0.2 for i in y_pos], df_plot['total_qty'], height=0.4, label='Quantité')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_plot.index, fontsize=9)
ax1.set_xlabel('Quantité vendue')
ax1_tw = ax1.twiny()
bars2 = ax1_tw.barh([i+0.2 for i in y_pos], df_plot['total_rev'], height=0.4, label='CA (€)')
ax1_tw.set_xlabel('Chiffre d\'affaires (€)')
# pie
data_rev = df_plot['total_rev'].sort_values(ascending=False)
ax2.pie(data_rev.values, labels=data_rev.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white'})
ax2.axis('equal')
plt.tight_layout()
st.pyplot(fig_prod)

# 2) Top aisles : dual-axe Quantité vs CA et camembert
st.subheader("Top rayons : Quantité vs Chiffre d'affaires")
aisle_qty = df.groupby('aisle')['order_id'].count().rename('total_qty')
aisle_rev = df.groupby('aisle')['price'].sum().rename('total_rev')
top_aisles = pd.concat([aisle_qty, aisle_rev], axis=1).nlargest(12, 'total_qty')
fig_aisle, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios':[2,1]})
y2 = range(len(top_aisles))
df_a = top_aisles.sort_values('total_qty')
b1 = ax3.barh([i-0.2 for i in y2], df_a['total_qty'], height=0.4, label='Quantité')
ax3.set_yticks(y2); ax3.set_yticklabels(df_a.index)
ax3.set_xlabel('Quantité vendue')
ax3_tw = ax3.twiny()
b2 = ax3_tw.barh([i+0.2 for i in y2], df_a['total_rev'], height=0.4, label='CA (€)')
ax3_tw.set_xlabel('CA (€)')
labels = [h.get_label() for h in [b1, b2]]
ax3.legend([b1, b2], labels, loc='lower right')
data_a = df_a['total_rev'].sort_values(ascending=False)
ax4.pie(data_a.values, labels=data_a.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white'})
ax4.axis('equal')
plt.tight_layout()
st.pyplot(fig_aisle)

# 3) Meilleurs moments : jour, heure et heatmap
st.subheader("Moments forts : jour et heure")
# Prépa
full = df.copy()
day_map = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']
top_days = full.groupby('order_dow')['product_id'].count()
top_days.index = [day_map[d] for d in top_days.index]
top_hours = full.groupby('order_hour_of_day')['product_id'].count()
sales_mat = full.groupby(['order_dow','order_hour_of_day']).size().unstack(fill_value=0)
sales_k = sales_mat/1000
# Figure
fig_mom, axes = plt.subplots(1, 3, figsize=(20,6), gridspec_kw={'width_ratios':[1,1,1.2]})
sns.barplot(x=top_days.index, y=top_days.values, palette='Greens', ax=axes[0])
axes[0].set(title='Produits commandés par jour', xlabel='Jour', ylabel='Nombre')
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(x=top_hours.index, y=top_hours.values, palette='Greens', ax=axes[1])
axes[1].set(title='Produits commandés par heure', xlabel='Heure', ylabel='Nombre')
axes[1].tick_params(axis='x', rotation=45)
cmap_rg = LinearSegmentedColormap.from_list('RG', ['white','green'])
sns.heatmap(sales_k, cmap=cmap_rg, annot=True, fmt='.0f', annot_kws={'fontsize':8}, cbar_kws={'label':'Commande (k)'}, linewidths=0.5, linecolor='white', ax=axes[2])
axes[2].set(title='Heatmap Jour x Heure (k)', xlabel='Heure', ylabel='Jour')
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
st.pyplot(fig_mom)

# 4) Méthode du coude pour RFM
st.subheader("Méthode du coude pour le clustering RFM")
# RFM par utilisateur
rfm = df[['user_id','Recency','Frequency','Monetary']].drop_duplicates()
X = StandardScaler().fit_transform(rfm[['Recency','Frequency','Monetary']])
inertia = []
ks = range(1,11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)
# Plot
fig_elbow = plt.figure(figsize=(8,4))
plt.plot(ks, inertia, marker='o')
plt.title('Elbow Method – Choix de K')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.grid(True)
st.pyplot(fig_elbow)

# 5) Pairplot RFM
st.subheader("Pairplot des variables RFM par cluster")
# Assure cluster existant
cluster_plot = df[['Recency','Frequency','Monetary','Cluster']].drop_duplicates()
g = sns.pairplot(cluster_plot, vars=['Recency','Frequency','Monetary'], hue='Cluster', palette='Set2')
st.pyplot(g.fig)

# 6) Scatter 3D Matplotlib
st.subheader("Scatter 3D Matplotlib des clusters RFM")
fig3d = plt.figure(figsize=(10,8))
ax = fig3d.add_subplot(111, projection='3d')
cols = cluster_plot['Cluster']
scatter = ax.scatter(cluster_plot['Recency'], cluster_plot['Frequency'], cluster_plot['Monetary'], c=cols, cmap='Set2', s=40, alpha=0.7)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('Clusters RFM 3D')
ax.legend(*scatter.legend_elements(), title='Cluster')
st.pyplot(fig3d)

# 7) Scatter 3D interactif Plotly
st.subheader("Scatter 3D Interactif Plotly")
fig_pl = px.scatter_3d(cluster_plot, x='Recency', y='Frequency', z='Monetary', color='Cluster', size_max=6, opacity=0.7, title='RFM Interactif')
st.plotly_chart(fig_pl, use_container_width=True)

# 8) WordCloud
st.subheader("WordCloud du texte d'article")
if os.path.exists('article.txt'):
    text = open('article.txt', 'r', encoding='utf-8').read()
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(wc.to_array(), use_column_width=True)
else:
    st.warning("article.txt non trouvé.")

# Footer
st.markdown("---")
st.caption("App Streamlit avec l'ensemble des visualisations du projet")
