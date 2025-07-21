import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import geopandas as gpd
from shapely.geometry import Point

# Import dataset
@st.cache_data
def load_data():
    df = pd.read_excel('C:/Users/HANIS/Downloads/cleaned_data.xlsx', sheet_name='cleaned_data')
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry) 
    return geo_df, df

geo_df, df = load_data()

# Clean Data
df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all')



# Columns for correlation analysis
cols = [
    'Renewable energy share in the total final energy consumption (%)',
    'Electricity from renewables (TWh)',
    'Low-carbon electricity (% electricity)',
    'Value_co2_emissions_kt_by_country',
    'gdp_per_capita',
    'Primary energy consumption per capita (kWh/person)',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
    'Land Area(Km2)'
]
renewable_df = df[cols].apply(pd.to_numeric, errors='coerce')
corr_matrix = renewable_df.corr()

sns.set(style="whitegrid", font_scale=1.1)

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Compute Renewable Potential Index
df['Renewable Potential Index'] = (
    (100 - df['Renewable energy share in the total final energy consumption (%)']) * 0.4 +
    (df['Land Area(Km2)'] / df['Land Area(Km2)'].max()) * 0.3 +
    (df['gdp_per_capita'] / df['gdp_per_capita'].max()) * 0.2 +
    (df['Value_co2_emissions_kt_by_country'] / df['Value_co2_emissions_kt_by_country'].max()) * 0.1
)

# Streamlit UI
st.set_page_config(layout="wide", page_title="Renewable Energy Investment Dashboard")
st.title("Renewable Energy Potential Assessment")
st.subheader("Identify regions for green investments based on renewable capacity")

# Sidebar filters
st.sidebar.header("Investment Filters")
years = sorted(df['Year'].unique())
year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1)
min_potential = st.sidebar.slider("Minimum Potential Index", 0.0, 1.0, 0.5)
max_energy_intensity = st.sidebar.slider(
    "Max Energy Intensity (MJ/$)", 
    min_value=0, 
    max_value=int(df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].max()), 
    value=5
)

# Filter data
filtered_df = df[
    (df['Year'] == year) &
    (df['Renewable Potential Index'] >= min_potential) &
    (df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'] <= max_energy_intensity)
]

# Key Metrics
st.header("Investment Opportunity Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("High Potential Countries", len(filtered_df))
col2.metric("Avg Renewable Potential", f"{filtered_df['Renewable Potential Index'].mean():.2%}")
col3.metric("Avg CO2 Emissions", f"{filtered_df['Value_co2_emissions_kt_by_country'].mean():,.0f} kt")
col4.metric("Avg Energy Intensity", f"{filtered_df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].mean():.1f} MJ/$")

# Map Visualization
st.header("Global Renewable Investment Potential")
st.caption("Size = Renewable electricity output, Color = Investment potential")

fig_map = px.scatter_geo(
    filtered_df,
    lat='Latitude',
    lon='Longitude',
    size='Electricity from renewables (TWh)',
    color='Renewable Potential Index',
    hover_name='Entity',
    projection='natural earth',
    color_continuous_scale='RdYlGn_r',
    size_max=30,
    scope='world'
)
fig_map.update_layout(height=600)
st.plotly_chart(fig_map, use_container_width=True)

# Correlation Analysis Tabs
st.header("Key Investment Correlations")
tab1, tab2, tab3 = st.tabs(["Renewable vs Economy", "Emissions Analysis", "Land Utilization"])

with tab1:
    fig = px.scatter(
        filtered_df,
        x='gdp_per_capita',
        y='Renewable energy share in the total final energy consumption (%)',
        size='Value_co2_emissions_kt_by_country',
        color='Renewable Potential Index',
        hover_name='Entity',
        log_x=True,
        labels={
            'gdp_per_capita': 'GDP per Capita (log scale)',
            'Renewable energy share in the total final energy consumption (%)': 'Renewable Share (%)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(
        filtered_df,
        x='Value_co2_emissions_kt_by_country',
        y='Low-carbon electricity (% electricity)',
        size='Land Area(Km2)',
        color='Renewable Potential Index',
        hover_name='Entity',
        log_x=True,
        labels={
            'Value_co2_emissions_kt_by_country': 'CO2 Emissions (kt, log scale)',
            'Low-carbon electricity (% electricity)': 'Low-Carbon Electricity (%)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    filtered_df['Renewable TWh per 1000 km²'] = filtered_df['Electricity from renewables (TWh)'] / (filtered_df['Land Area(Km2)'] / 1000)
    fig = px.scatter(
        filtered_df,
        x='Land Area(Km2)',
        y='Renewable TWh per 1000 km²',
        size='gdp_per_capita',
        color='Renewable Potential Index',
        hover_name='Entity',
        log_x=True,
        labels={
            'Land Area(Km2)': 'Land Area (km², log scale)',
            'Renewable TWh per 1000 km²': 'Renewable Output Efficiency (TWh/1000 km²)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# Top Investment Targets
st.header("Top Investment Opportunities")
filtered_df = filtered_df.sort_values('Renewable Potential Index', ascending=False)
st.dataframe(
    filtered_df[['Entity', 'Renewable Potential Index', 
                'Renewable energy share in the total final energy consumption (%)',
                'Electricity from renewables (TWh)', 'Land Area(Km2)',
                'gdp_per_capita']].rename(columns={
        'Entity': 'Country',
        'Renewable Potential Index': 'Potential Index',
        'Renewable energy share in the total final energy consumption (%)': 'Renewable Share (%)',
        'Electricity from renewables (TWh)': 'Renewable Output (TWh)',
        'Land Area(Km2)': 'Land Area (km²)',
        'gdp_per_capita': 'GDP per Capita'
    }).head(20),
    height=400,
    use_container_width=True
)

# Trend Analysis
st.header("Renewable Energy Adoption Trends")
top_5 = filtered_df['Entity'].head(5).tolist()
trend_df = df[df['Entity'].isin(top_5)]

fig = px.line(
    trend_df,
    x='Year',
    y='Renewable energy share in the total final energy consumption (%)',
    color='Entity',
    markers=True,
    title='Renewable Share Growth of Top Investment Targets',
    labels={'Renewable energy share in the total final energy consumption (%)': 'Renewable Share (%)'}
)
st.plotly_chart(fig, use_container_width=True)

# Optional heatmap (Plotly)
corr_values = corr_matrix.values.round(2)
x_labels = list(corr_matrix.columns)
y_labels = list(corr_matrix.index)

fig_corr = ff.create_annotated_heatmap(
    z=corr_values,
    x=x_labels,
    y=y_labels,
    colorscale='RdBu',
    showscale=True,
    reversescale=True,
    zmin=-1,
    zmax=1,
    annotation_text=corr_values,
    hoverinfo="z"
)

fig_corr.update_layout(
    title='Correlation Heatmap of Key Indicators',
    width=900,
    height=700
)
st.plotly_chart(fig_corr, use_container_width=True)


# Identify promising countries based on stricter investment criteria
st.header("🌟 Top Recommended Investment Destinations")

# Ensure 'Investment Score' exists or assign it from Renewable Potential Index
df['Investment Score'] = df['Renewable Potential Index']
latest_df = df[df['Year'] == year]

# Apply the criteria
promising_countries = latest_df[
    (latest_df['Investment Score'] > 0.7) &
    (latest_df['gdp_per_capita'] > 3000) &
    (latest_df['Renewable energy share in the total final energy consumption (%)'] > 20)
].sort_values('Investment Score', ascending=False)

if not promising_countries.empty:
    st.success("**Top Recommended Investment Destinations:**")
    cols = st.columns(3)
    for i, (_, row) in enumerate(promising_countries.head(6).iterrows()):
        with cols[i % 3]:
            st.metric(
                label=row['Entity'],
                value=f"Score: {row['Investment Score']:.2f}",
                help=f"""
                Renewable Share: {row['Renewable energy share in the total final energy consumption (%)']:.1f}%
                GDP per capita: ${row['gdp_per_capita']:,.0f}
                CO₂ Emissions: {row['Value_co2_emissions_kt_by_country']:,.0f} kt
                """
            )
else:
    st.warning("⚠️ No countries meet the recommended investment criteria for the selected year and filters.")

# Methodology
with st.expander("Methodology"):
    st.markdown("""
    **Renewable Potential Index**:
    - Based on four factors:
      1. (100 - Renewable Share) × 40%
      2. (Land Area / Max Land Area) × 30%
      3. (GDP per Capita / Max GDP) × 20%
      4. (CO2 Emissions / Max CO2) × 10%

    **Prioritization Targets**:
    - Countries with:
      - Low current renewable adoption
      - High land availability
      - Strong economy
      - High emissions

    **Source**: Global Energy Dataset (cleaned_data.xlsx)
    """)

