import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Page setup
st.set_page_config(page_title="Green Investment Explorer", layout="wide")
st.title("🌍 Renewable Energy Investment Dashboard")

# File uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload your cleaned_data.xlsx", type=["xlsx"])

@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name='cleaned_data')
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    return geo_df, df

if uploaded_file is not None:
    geo_df, df = load_data(uploaded_file)

    # Show raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.write(df)

    # Correlation heatmap
    st.subheader("🔍 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_cols.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatter plot: Investment Score vs GDP per Capita
    st.subheader("📈 Investment Score vs GDP per Capita")
    fig2 = px.scatter(
        df,
        x="gdp_per_capita",
        y="Investment Score",
        size="Access to electricity (% of population)",
        color="Renewable energy share in the total final energy consumption (%)",
        hover_name="Entity",
        size_max=40
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Top countries recommendation
    st.subheader("💡 Top Recommended Investment Destinations")
    promising_countries = df[
        (df['Investment Score'] > 0.7) &
        (df['gdp_per_capita'] > 3000) &
        (df['Renewable energy share in the total final energy consumption (%)'] > 20)
    ].sort_values('Investment Score', ascending=False)

    if not promising_countries.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(promising_countries.head(6).iterrows()):
            with cols[i % 3]:
                st.metric(
                    label=row['Entity'],
                    value=f"Score: {row['Investment Score']:.2f}",
                    help=f"""
                    Renewable Share: {row['Renewable energy share in the total final energy consumption (%)']:.1f}%
                    Electricity Access: {row['Access to electricity (% of population)']:.1f}%
                    GDP per capita: ${row['gdp_per_capita']:,.0f}
                    """
                )
    else:
        st.warning("⚠️ No countries meet the investment criteria.")

else:
    st.warning("Please upload the `cleaned_data.xlsx` file to continue.")
