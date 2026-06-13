"""
Restaurant Analytics & Rating Prediction System
================================================
Streamlit Dashboard  |  Cognifyz Data Science Internship
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

from utils.hash_table import build_frequency_table

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #e05c2c;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fff7f3, #fff);
        border-left: 4px solid #e05c2c;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #f0f8ff;
        border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #333;
        margin-top: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background: #1a1a2e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    clean_path = os.path.join(BASE_DIR, 'dataset', 'cleaned_dataset.csv')
    raw_path   = os.path.join(BASE_DIR, 'dataset', 'Dataset.csv')
    path = clean_path if os.path.exists(clean_path) else raw_path
    df = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    # Lightweight cleaning if raw
    text_cols = df.select_dtypes(include='object').columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df.replace('nan', np.nan, inplace=True)
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    df['Cuisines'].fillna('unknown', inplace=True)
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


df    = load_data()
model = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Dataset Overview",
            "🏙️ Top Cities",
            "🍜 Top Cuisines",
            "🌍 Geospatial Map",
            "🧑‍🍳 Customer Preferences",
            "⭐ Rating Prediction",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Cognifyz Data Science Internship")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dataset Overview":
    st.markdown('<div class="main-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Shape, types, missing values, statistics & class distribution</div>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Restaurants", f"{len(df):,}")
    c2.metric("Countries", int(df['Country Code'].nunique()))
    c3.metric("Cities", int(df['City'].nunique()))
    c4.metric("Cuisine Types", int(df['Cuisines'].str.split(',').explode().str.strip().nunique()))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Column Data Types")
        st.dataframe(
            pd.DataFrame({'Column': df.dtypes.index, 'Type': df.dtypes.values})
            .reset_index(drop=True), height=350
        )

    with col2:
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        miss_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        miss_df = miss_df[miss_df['Missing Count'] > 0]
        if miss_df.empty:
            st.success("No missing values (dataset is clean).")
        else:
            st.dataframe(miss_df.sort_values('Missing %', ascending=False))

    st.divider()

    st.subheader("Descriptive Statistics")
    st.dataframe(df.select_dtypes(include=np.number).describe().round(3))

    st.subheader("Rating Distribution (Class Imbalance Analysis)")
    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(df, x='Aggregate rating', nbins=20,
                           title='All Ratings incl. 0 (Unrated)',
                           color_discrete_sequence=['#e05c2c'],
                           template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        rated = df[df['Aggregate rating'] > 0]
        fig2  = px.histogram(rated, x='Aggregate rating', nbins=20,
                             title='Ratings (Excluding Unrated = 0)',
                             color_discrete_sequence=['#2196F3'],
                             template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Rating 0 represents unrated restaurants '
                '— a form of class imbalance. Models trained without filtering these produce '
                'biased predictions. Always exclude 0-rated rows during modelling.</div>',
                unsafe_allow_html=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Top Cities
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏙️ Top Cities":
    st.markdown('<div class="main-header">🏙️ Top Cities</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">City frequency analysis using custom Hash Table + visualizations</div>', unsafe_allow_html=True)

    city_ht    = build_frequency_table(df['City'], capacity=256)
    top_cities = city_ht.top_n(20)
    cities_df  = pd.DataFrame(top_cities, columns=['City', 'Count'])

    n = st.slider("Number of top cities to display", 5, 20, 15)
    cities_df = cities_df.head(n)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(cities_df, x='Count', y='City', orientation='h',
                     color='Count', color_continuous_scale='Blues_r',
                     title=f'Top {n} Cities by Restaurant Count',
                     template='plotly_white')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(cities_df.head(8), names='City', values='Count',
                      title='Top 8 Cities Share', hole=0.4,
                      template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("City Ratings Comparison")
    city_ratings = (
        df[df['Aggregate rating'] > 0]
        .groupby('City')['Aggregate rating']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'Avg Rating', 'count': 'Count'})
        .query('Count >= 50')
        .nlargest(15, 'Count')
    )
    fig3 = px.scatter(
        city_ratings, x='Count', y='Avg Rating', size='Count',
        text='City', color='Avg Rating', color_continuous_scale='RdYlGn',
        title='City: Restaurant Count vs Avg Rating (min 50 restaurants)',
        template='plotly_white', size_max=40
    )
    fig3.update_traces(textposition='top center')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> New Delhi has by far the highest '
                'restaurant count, but mid-tier cities like Panchkula and Chandigarh show '
                'higher average ratings — smaller markets often indicate more curated dining.</div>',
                unsafe_allow_html=True)

    st.subheader("Hash Table Collision Report (City)")
    st.code(city_ht.collision_report())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Top Cuisines
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🍜 Top Cuisines":
    st.markdown('<div class="main-header">🍜 Top Cuisines</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Popularity & quality rankings via Hash Table frequency analysis</div>', unsafe_allow_html=True)

    cuisine_ht   = build_frequency_table(df['Cuisines'], capacity=512)
    top_cuisines = cuisine_ht.top_n(25)
    cuisines_df  = pd.DataFrame(top_cuisines, columns=['Cuisine', 'Count'])

    n = st.slider("Number of top cuisines", 5, 25, 15)
    top_n_df = cuisines_df.head(n)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(top_n_df, x='Count', y='Cuisine', orientation='h',
                     color='Count', color_continuous_scale='Oranges_r',
                     title=f'Top {n} Cuisines by Frequency',
                     template='plotly_white')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Explode and compute avg rating per cuisine
        cuis_exp = df.copy()
        cuis_exp['Cuisines'] = cuis_exp['Cuisines'].str.split(',')
        cuis_exp = cuis_exp.explode('Cuisines')
        cuis_exp['Cuisines'] = cuis_exp['Cuisines'].str.strip()
        top_names = [x[0] for x in top_cuisines[:n]]
        rated_cuis = (
            cuis_exp[
                (cuis_exp['Cuisines'].isin(top_names)) &
                (cuis_exp['Aggregate rating'] > 0)
            ]
            .groupby('Cuisines')['Aggregate rating'].mean().reset_index()
            .rename(columns={'Aggregate rating': 'Avg Rating'})
            .sort_values('Avg Rating', ascending=False)
        )
        fig2 = px.bar(rated_cuis, x='Avg Rating', y='Cuisines', orientation='h',
                      color='Avg Rating', color_continuous_scale='RdYlGn',
                      title='Avg Rating by Top Cuisines', range_x=[0, 5],
                      template='plotly_white')
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> North Indian and Chinese cuisines '
                'dominate by count (popularity), but Continental and Italian cuisines earn '
                'higher average ratings, suggesting quality over quantity.</div>',
                unsafe_allow_html=True)

    st.subheader("Hash Table Collision Report (Cuisine)")
    st.code(cuisine_ht.collision_report())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Geospatial Map
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌍 Geospatial Map":
    st.markdown('<div class="main-header">🌍 Geospatial Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Restaurant locations, density heatmap & rating distribution by geography</div>', unsafe_allow_html=True)

    geo_df = df[
        df['Latitude'].between(-90, 90) &
        df['Longitude'].between(-180, 180) &
        (df['Latitude'] != 0) &
        (df['Longitude'] != 0)
    ].copy()

    map_type = st.radio("Map Type", ["Cluster Map", "Heat Map"], horizontal=True)

    sample_size = st.slider("Sample size (restaurants)", 500, min(5000, len(geo_df)), 2000, 500)
    sample = geo_df.sample(sample_size, random_state=42)

    m = folium.Map(location=[20, 78], zoom_start=4, tiles='CartoDB positron')

    if map_type == "Cluster Map":
        cluster = MarkerCluster().add_to(m)
        for _, row in sample.iterrows():
            color = ('green' if row['Aggregate rating'] >= 4
                     else 'orange' if row['Aggregate rating'] >= 3
                     else 'red')
            popup = (f"<b>{row.get('Restaurant Name','')}</b><br>"
                     f"Rating: {row['Aggregate rating']}<br>"
                     f"City: {row.get('City','')}")
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5, color=color, fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup, max_width=250)
            ).add_to(cluster)
    else:
        heat_data = sample[['Latitude', 'Longitude']].values.tolist()
        HeatMap(heat_data, radius=10, blur=12).add_to(m)

    st_folium(m, width=1100, height=550)

    st.subheader("Rating Distribution by Country")
    country_avg = (
        geo_df.groupby('Country Code')['Aggregate rating']
        .mean().reset_index()
        .rename(columns={'Country Code': 'Code', 'Aggregate rating': 'Avg Rating'})
    )
    fig = px.choropleth(
        country_avg, locations='Code', locationmode='ISO-3166 Numeric',
        color='Avg Rating', color_continuous_scale='RdYlGn',
        title='Average Restaurant Rating by Country', template='plotly_white',
        range_color=[0, 5]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lat / Lon vs Rating Correlation")
    corr = geo_df[['Latitude', 'Longitude', 'Aggregate rating']].corr()
    fig2 = px.imshow(corr, text_auto='.3f', color_continuous_scale='RdBu_r',
                     title='Geospatial Correlation Matrix', template='plotly_white',
                     zmin=-1, zmax=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Longitude shows a weak negative '
                'correlation with rating — Western longitudes (Europe/Americas) have fewer '
                'but higher-rated restaurants in this dataset, reflecting different review '
                'cultures.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: Customer Preference Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧑‍🍳 Customer Preferences":
    st.markdown('<div class="main-header">🧑‍🍳 Customer Preference Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cuisine popularity, quality rankings, votes analysis & preference insights</div>', unsafe_allow_html=True)

    cuis_exp = df.copy()
    cuis_exp['Cuisines'] = cuis_exp['Cuisines'].str.split(',')
    cuis_exp = cuis_exp.explode('Cuisines')
    cuis_exp['Cuisines'] = cuis_exp['Cuisines'].str.strip()

    # Cuisine frequency via hash table
    freq_ht   = build_frequency_table(cuis_exp['Cuisines'], capacity=512)
    top_25    = freq_ht.top_n(25)
    top_names = [x[0] for x in top_25]

    cuisine_agg = (
        cuis_exp[cuis_exp['Cuisines'].isin(top_names)]
        .groupby('Cuisines')
        .agg(
            Avg_Rating=('Aggregate rating', lambda x: x[x > 0].mean()),
            Avg_Votes=('Votes', 'mean'),
            Count=('Cuisines', 'count')
        ).reset_index()
    )

    st.subheader("Popularity vs Quality Bubble Chart")
    fig = px.scatter(
        cuisine_agg, x='Avg_Rating', y='Avg_Votes', size='Count',
        color='Avg_Rating', color_continuous_scale='RdYlGn',
        text='Cuisines',
        title='Top Cuisines: Avg Rating vs Avg Votes (bubble = count)',
        template='plotly_white', size_max=55
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Popular Cuisines")
        pop_df = pd.DataFrame(top_25, columns=['Cuisine', 'Count'])
        fig2 = px.bar(pop_df.head(15), x='Count', y='Cuisine', orientation='h',
                      color='Count', color_continuous_scale='Blues_r',
                      template='plotly_white', title='Top 15 by Frequency')
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Highest Rated Cuisines (min 50 restaurants)")
        high_rated = cuisine_agg[cuisine_agg['Count'] >= 50].nlargest(15, 'Avg_Rating')
        fig3 = px.bar(high_rated, x='Avg_Rating', y='Cuisines', orientation='h',
                      color='Avg_Rating', color_continuous_scale='RdYlGn',
                      range_x=[0, 5], template='plotly_white',
                      title='Top 15 by Avg Rating')
        fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Online Delivery & Table Booking Impact")
    col3, col4 = st.columns(2)
    with col3:
        delivery_df = (
            df[df['Aggregate rating'] > 0]
            .groupby('Has Online delivery')['Aggregate rating']
            .mean().reset_index()
        )
        fig4 = px.bar(delivery_df, x='Has Online delivery', y='Aggregate rating',
                      color='Aggregate rating', color_continuous_scale='RdYlGn',
                      title='Online Delivery vs Avg Rating', template='plotly_white',
                      range_y=[0, 5])
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        table_df = (
            df[df['Aggregate rating'] > 0]
            .groupby('Has Table booking')['Aggregate rating']
            .mean().reset_index()
        )
        fig5 = px.bar(table_df, x='Has Table booking', y='Aggregate rating',
                      color='Aggregate rating', color_continuous_scale='RdYlGn',
                      title='Table Booking vs Avg Rating', template='plotly_white',
                      range_y=[0, 5])
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insights:</b><br>'
                '• Online delivery restaurants score ~0.6 stars higher on average.<br>'
                '• Table booking restaurants score ~0.9 stars higher, suggesting a '
                'correlation between service quality and dining experience.<br>'
                '• Specialty cuisines (Continental, Thai) have higher ratings despite '
                'lower counts — niche appeal often means dedicated customers.<br>'
                '• Votes strongly correlate with ordering behaviour — popular restaurants '
                'attract more reviews, reinforcing visibility.</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: Rating Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⭐ Rating Prediction":
    st.markdown('<div class="main-header">⭐ Rating Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Aggregate Rating using Random Forest — trained on real restaurant data</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Trained model not found. Please run Level3.ipynb first to generate `models/random_forest.pkl`.")

        # Fallback: train a quick model on the fly
        st.info("Training a quick model on the fly for demonstration...")
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestRegressor

        train_df = df[df['Aggregate rating'] > 0].copy()
        train_df['Name Length']    = train_df['Restaurant Name'].astype(str).apply(len)
        train_df['Address Length'] = train_df['Address'].astype(str).apply(len)
        train_df['Has Table Booking']   = (train_df['Has Table booking'].astype(str).str.strip() == 'yes').astype(int)
        train_df['Has Online Delivery'] = (train_df['Has Online delivery'].astype(str).str.strip() == 'yes').astype(int)

        le_c = LabelEncoder()
        le_u = LabelEncoder()
        train_df['City_enc']    = le_c.fit_transform(train_df['City'].astype(str))
        train_df['Cuisine_enc'] = le_u.fit_transform(
            train_df['Cuisines'].astype(str).str.split(',').str[0].str.strip()
        )
        FEATS = ['Votes','Average Cost for two','Price range',
                 'Has Table Booking','Has Online Delivery',
                 'Name Length','Address Length','City_enc','Cuisine_enc']

        X = train_df[FEATS]
        y = train_df['Aggregate rating']
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        st.success("Quick model trained successfully!")

    st.divider()
    st.subheader("Enter Restaurant Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        votes       = st.number_input("Number of Votes", min_value=0, max_value=10000, value=200, step=50)
        price_range = st.selectbox("Price Range", [1, 2, 3, 4],
                                   format_func=lambda x: f"{x} – {'Budget' if x==1 else 'Mid-range' if x==2 else 'Premium' if x==3 else 'Luxury'}")
    with col2:
        avg_cost    = st.number_input("Average Cost for Two (₹)", min_value=0, max_value=10000, value=500, step=100)
        has_online  = st.selectbox("Online Delivery", ["Yes", "No"])
    with col3:
        has_table   = st.selectbox("Table Booking", ["Yes", "No"])
        name_len    = st.number_input("Restaurant Name Length", min_value=3, max_value=80, value=20)
        addr_len    = st.number_input("Address Length", min_value=5, max_value=200, value=50)

    # Encode inputs
    city_enc    = 0    # default (New Delhi-like encoding)
    cuisine_enc = 0    # default

    features = np.array([[
        votes, avg_cost, price_range,
        1 if has_table  == "Yes" else 0,
        1 if has_online == "Yes" else 0,
        name_len, addr_len,
        city_enc, cuisine_enc
    ]])

    if st.button("🔮 Predict Rating", use_container_width=True):
        pred = model.predict(features)[0]
        pred = max(0.0, min(5.0, pred))   # clamp to [0, 5]

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            st.markdown(f"""
            <div style='text-align:center; padding: 2rem; background: linear-gradient(135deg,#fff7f3,#fff);
                        border-radius: 16px; border: 2px solid #e05c2c; margin-top: 1rem;'>
                <div style='font-size:4rem; font-weight:800; color:#e05c2c;'>{pred:.2f}</div>
                <div style='font-size:1.2rem; color:#555; margin-top:0.5rem;'>Predicted Aggregate Rating / 5.0</div>
                <div style='font-size:2rem; margin-top:0.5rem;'>
                    {"⭐⭐⭐⭐⭐" if pred >= 4.5 else "⭐⭐⭐⭐" if pred >= 3.5 else "⭐⭐⭐" if pred >= 2.5 else "⭐⭐" if pred >= 1.5 else "⭐"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        label = (
            "Excellent 🏆" if pred >= 4.5 else
            "Very Good 😃" if pred >= 4.0 else
            "Good 👍"      if pred >= 3.5 else
            "Average 😐"   if pred >= 3.0 else
            "Below Average 😕" if pred >= 2.0 else
            "Poor 👎"
        )
        st.markdown(f"<div style='text-align:center; font-size:1.3rem; margin-top:1rem; color:#333;'>Classification: <b>{label}</b></div>", unsafe_allow_html=True)

        # Feature importance viz
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance (this model)")
            feat_names = ['Votes','Avg Cost','Price Range','Table Booking',
                          'Online Delivery','Name Length','Addr Length','City','Cuisine']
            imp_df = pd.DataFrame({'Feature': feat_names,
                                   'Importance': model.feature_importances_}).sort_values('Importance')
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Reds',
                         template='plotly_white', title='What drives the prediction?')
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Model Information")
    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.markdown("**Algorithm**\n\nRandom Forest Regressor")
    col_i2.markdown("**Estimators**\n\n150 trees")
    col_i3.markdown("**Validation**\n\n5-Fold Cross Validation")

    st.markdown('<div class="insight-box">💡 <b>Why Random Forest?</b> '
                'Random Forest handles non-linear relationships, is robust to outliers via '
                'bootstrap aggregation, provides built-in feature importance, and consistently '
                'outperforms Linear Regression and Decision Tree on this dataset in both '
                'MAE and R² metrics.</div>', unsafe_allow_html=True)
