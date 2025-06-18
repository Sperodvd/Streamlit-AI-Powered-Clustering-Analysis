import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("AI-Powered Clustering & Insight Analysis")

st.markdown("""
## Overview

This application enables users to explore, visualize, and analyze datasets using advanced clustering techniques and LLM-powered insights. 
The app is designed to serve as an initial phases of data analysis, providing users with a comprehensive initial view of their datasets. 


---

### Key Features

- **Flexible Data Input:**  
  Upload your own CSV files or select from built-in sample datasets (Mall Customers Segmentation, Wine Dataset Clustering) for instant analysis.

- **Data Preview & Cleaning:**  
  Preview the first few rows of the dataset and display column information. Automatically detect missing values and choose how to handle them: drop rows/columns, fill with zeros or means, or leave as is.

- **Interactive Visualizations:**  
  Generate a variety of visualizations, including:
  - Pairplots for exploring relationships between variables
  - Correlation heatmaps for understanding feature interdependencies
  - Histograms (with multi-column support and subplot arrangement) for distribution analysis  
  Two visualizations can be displayed side-by-side for easy comparison.

- **Clustering Analysis:**  
  - **Variable Selection:** Choose which numeric variables to include or exclude from clustering.
  - **Elbow Method & KElbow Visualizer:** Determine the optimal number of clusters using both inertia plots and Yellowbrick’s KElbow Visualizer.
  - **Clustering Execution:** Run K-Means clustering and visualize the results in 2D using PCA.
  - **Cluster Profiles:** View the average feature values for each cluster.
  - **Silhouette Score:** Evaluate clustering quality with this metric.

- **AI-Powered Summarization:**  
  After clustering, a dedicated Data Scientist AI-Agent, powered by Groq API (utilizing the Llama 3 8B LLM), will analyze cluster profiles and silhouette scores and interprets the key insights, presenting the findings in a clear and accessible manner for the user. The app caches LLM responses to avoid redundant API calls.

---

### How to Use

1. **Select or Upload Data:** Choose a sample dataset or upload your own CSV.
2. **Preview & Clean Data:** Review the data and handle missing values as needed.
3. **Visualize:** Select and compare different visualizations to explore your data.
4. **Cluster:** Run the clustering workflow, analyze results, and view cluster profiles.
5. **Summarize:** Click the summarization button to receive an AI-generated interpretation of your clustering results.

---

**Developed for educational and exploratory data science purposes.**  
*Empowering users to discover insights in their data with the help of AI.*
""")