import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Define a consistent color palette for all visualizations
distinct_colors = [
    '#E41A1C',  # Bright Red
    '#377EB8',  # Strong Blue
    '#4DAF4A',  # Bright Green
    '#984EA3',  # Purple
    '#FF7F00',  # Orange
    '#FFFF33',  # Yellow
    '#A65628',  # Brown
    '#F781BF',  # Pink
    '#1B9E77',  # Teal
    '#D95F02'  # Dark Orange
]


# --- Helper Functions ---
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def preprocess_data(df, selected_features):
    # Handle missing values
    for col in selected_features:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Outlier removal using IQR
    for col in selected_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def scale_features(df, selected_features):
    scaler = MinMaxScaler()
    df[selected_features] = scaler.fit_transform(df[selected_features])
    return df


def determine_optimal_clusters(data, selected_features):
    silhouette_scores = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data[selected_features])
        score = silhouette_score(data[selected_features], labels)
        silhouette_scores[k] = score

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    return optimal_k, silhouette_scores


def perform_clustering(data, selected_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[selected_features])
    return data, kmeans


def create_cluster_visualizations(data, selected_features):
    # Perform PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data[selected_features])
    data['PCA1'], data['PCA2'] = reduced_data[:, 0], reduced_data[:, 1]

    # Calculate centroids
    centroids = data.groupby('Cluster')[['PCA1', 'PCA2']].mean().reset_index()

    n_clusters = len(data['Cluster'].unique())
    colors = distinct_colors[:n_clusters]

    # Create scatter plot with new colors
    fig = px.scatter(data, x='PCA1', y='PCA2', color='Cluster',
                     title='2D PCA Cluster Visualization',
                     labels={'PCA1': 'Principal Component 1',
                             'PCA2': 'Principal Component 2'},
                     hover_data=selected_features,
                     color_discrete_sequence=colors)

    # Add centroids with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=centroids['PCA1'],
            y=centroids['PCA2'],
            mode='markers',
            marker=dict(
                symbol='star',
                size=25,  # Increased size
                color='black',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            hovertext=[f'Centroid {i}' for i in centroids['Cluster']],
            showlegend=True
        )
    )

    # Update layout with improved visibility
    fig.update_layout(
        legend_title_text='Clusters',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, l=50, r=50, b=50),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='Black',
            borderwidth=1
        )
    )

    # Lighter grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')

    # Enhance marker visibility
    fig.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color='white')),
        selector=dict(mode='markers')
    )

    st.plotly_chart(fig)


def plot_correlation_heatmap(data, selected_features):
    correlation_matrix = data[selected_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(plt)


def plot_dendrogram(data, selected_features):
    linked = linkage(data[selected_features], 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    st.pyplot(plt)


def plot_cluster_size_pie_chart(data):
    cluster_counts = data['Cluster'].value_counts()
    n_clusters = len(cluster_counts)
    colors = distinct_colors[:n_clusters]

    fig = px.pie(values=cluster_counts, names=cluster_counts.index,
                 title="Cluster Size Distribution",
                 color_discrete_sequence=colors)

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        legend_title_text='Clusters',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig)


def plot_pairwise_distribution(data, selected_features):
    pairwise_data = data[selected_features + ['Cluster']]
    n_clusters = len(data['Cluster'].unique())
    colors = distinct_colors[:n_clusters]

    fig = px.scatter_matrix(
        pairwise_data,
        dimensions=selected_features,
        color='Cluster',
        title="Pairwise Feature Distribution per Cluster",
        color_discrete_sequence=colors
    )

    fig.update_traces(diagonal_visible=False)
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig)


def create_3d_cluster_visualization(data, selected_features):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data[selected_features])
    data['PCA1'], data['PCA2'], data['PCA3'] = reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2]

    n_clusters = len(data['Cluster'].unique())
    colors = distinct_colors[:n_clusters]

    fig = px.scatter_3d(data, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                        title='3D PCA Cluster Visualization',
                        labels={'PCA1': 'PC1', 'PCA2': 'PC2', 'PCA3': 'PC3'},
                        color_discrete_sequence=colors)

    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="white"),
            yaxis=dict(backgroundcolor="white"),
            zaxis=dict(backgroundcolor="white")
        )
    )
    st.plotly_chart(fig)


def plot_cluster_profiles(data, selected_features):
    n_clusters = len(data['Cluster'].unique())
    colors = distinct_colors[:n_clusters]

    for feature in selected_features:
        fig = px.box(data, x='Cluster', y=feature,
                     title=f"Cluster Profile: {feature}",
                     color='Cluster',
                     color_discrete_sequence=colors)

        fig.update_layout(
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig)


def plot_cluster_characteristics_radar(data, selected_features):
    cluster_means = data.groupby('Cluster')[selected_features].mean()
    fig = go.Figure()

    for i, cluster in enumerate(cluster_means.index):
        fig.add_trace(go.Scatterpolar(
            r=cluster_means.loc[cluster].values,
            theta=selected_features,
            fill='toself',
            name=f'Cluster {cluster}',
            line=dict(color=distinct_colors[i])
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showline=True)),
        showlegend=True,
        title="Cluster Characteristics Radar Chart",
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    st.plotly_chart(fig)


def plot_distance_to_centroid(data, kmeans, selected_features):
    distances = kmeans.transform(data[selected_features]).min(axis=1)
    cluster_distances = pd.DataFrame({
        'Distance': distances,
        'Cluster': data['Cluster']
    })

    fig = px.histogram(
        cluster_distances,
        x='Distance',
        color='Cluster',
        title="Distance to Cluster Centroid",
        color_discrete_sequence=distinct_colors
    )

    fig.update_layout(
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig)


def plot_elbow_method(data, selected_features):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[selected_features])
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    st.pyplot(fig)


# --- Streamlit App ---
st.title("Advanced Customer Segmentation")
st.sidebar.title("Upload & Configure")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        st.sidebar.success("File uploaded successfully!")
        st.dataframe(data.head())

        # Feature Selection
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_features = st.sidebar.multiselect("Select features for clustering", numeric_cols,
                                                   default=numeric_cols[:3])

        if len(selected_features) < 2:
            st.error("Please select at least two features for clustering.")
        else:
            # Preprocess Data
            data = preprocess_data(data, selected_features)
            data = scale_features(data, selected_features)

            # Optimal Clusters
            st.sidebar.subheader("Cluster Configuration")
            optimal_k, silhouette_scores = determine_optimal_clusters(data, selected_features)
            st.sidebar.write(f"Optimal number of clusters (Silhouette Score): {optimal_k}")

            # Elbow Method
            st.header("Elbow Method")
            plot_elbow_method(data, selected_features)

            # Cluster Selection
            n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=optimal_k)

            # Perform Clustering
            data, kmeans = perform_clustering(data, selected_features, n_clusters)

            # Visualization
            st.header("Cluster Visualizations")
            create_cluster_visualizations(data, selected_features)

            # Radar Chart for Cluster Analysis
            plot_cluster_characteristics_radar(data, selected_features)

            # Cluster Size Pie Chart
            plot_cluster_size_pie_chart(data)

            # Pairwise Feature Distribution
            plot_pairwise_distribution(data, selected_features)

            # 3D PCA Cluster Visualization
            create_3d_cluster_visualization(data, selected_features)

            # Correlation Heatmap
            st.header("Feature Correlation")
            plot_correlation_heatmap(data, selected_features)

            # Silhouette Score
            st.header("Clustering Quality")
            silhouette_scores_for_samples = silhouette_score(data[selected_features], data['Cluster'])
            st.write(f"Silhouette Score: {silhouette_scores_for_samples:.3f}")

            # Cluster Profile Boxplots
            st.header("Cluster Profiles")
            plot_cluster_profiles(data, selected_features)

            # Dendrogram
            st.header("Hierarchical Clustering")
            plot_dendrogram(data, selected_features)

            # Distance to Centroid Distribution
            st.header("Cluster Cohesion")
            plot_distance_to_centroid(data, kmeans, selected_features)

            # Display Cluster Metrics
            st.header("Cluster Metrics")
            cluster_summary = data.groupby('Cluster')[selected_features].mean()
            st.dataframe(cluster_summary)
else:
    st.info("Please upload a CSV file to start.")