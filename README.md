# Advanced Customer Segmentation

This project performs advanced customer segmentation using clustering techniques on a dataset of mall customers. The app leverages KMeans clustering and visualizations to uncover patterns in customer data and provide insights for targeted marketing or strategic business decisions.

## Features

- **File Upload**: Users can upload a CSV file containing customer data.
- **Data Preprocessing**: Handles missing values and removes outliers.
- **Clustering**: Uses KMeans clustering to segment customers based on selected features.
- **Optimal Clusters Detection**: Uses the Silhouette Score and the Elbow Method to suggest the optimal number of clusters.
- **Visualizations**:
  - **PCA-based 2D & 3D Cluster Visualizations**
  - **Cluster Size Pie Chart**
  - **Correlation Heatmap**
  - **Cluster Profiles (Boxplots & Radar Charts)**
  - **Dendrogram for Hierarchical Clustering**
  - **Pairwise Feature Distribution**
  - **Distance to Centroid Distribution**
- **Cluster Quality Metrics**: Computes and displays the Silhouette Score for clustering quality.
- **Cluster Metrics**: Displays mean values for selected features within each cluster.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Numpy
- Plotly
- Seaborn
- Matplotlib
- Scikit-learn
- SciPy

You can install all required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the app with Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Upload your CSV file containing customer data.
3. Select features to cluster on, and the app will display visualizations and metrics based on the clustering.

## Example Data Format

The dataset should include columns such as:

- `CustomerID`: Unique identifier for each customer
- `Gender`: Customer's gender
- `Age`: Customer's age
- `Annual Income (k$)`: Customer's annual income in thousands
- `Spending Score (1-100)`: Customer's spending score (1 to 100)

Example:
```csv
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
...
```

## Acknowledgments

- The project uses KMeans clustering from scikit-learn for customer segmentation.
- Plotly, Matplotlib, and Seaborn are used for creating interactive and static visualizations.
- The hierarchical clustering dendrogram is generated using SciPy's linkage function.

---

This README provides a basic overview and usage instructions for the project. Feel free to modify or expand it based on additional details you'd like to include.
