import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from groq import Groq
import os

#groq setup
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

st.set_page_config(layout="wide")
st.title("AI-Powered Clustering & Insight Analysis")

with st.sidebar:
    data_option = st.radio(
        "Choose data source:",
        ("Upload CSV file", "Use sample data")
    )

    uploaded_file = None
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    elif data_option == "Use sample data":
        sample_files = {
            "Mall Customers Segmentation": "Mall_Customers_Segmentation.csv",
            "Wine Dataset Clustering": "Wine_Dataset_Clustering.csv"
        }
        sample_choice = st.selectbox(
            "Select a sample dataset:",
            list(sample_files.keys())
        )
        sample_path = os.path.join("data", sample_files[sample_choice])
        if os.path.exists(sample_path):
            st.success(f"Sample data selected: {sample_files[sample_choice]}")
            uploaded_file = sample_path
        else:
            st.error(f"Sample file {sample_files[sample_choice]} not found in data folder.")

if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.subheader("Preview of the DataFrame")
    st.dataframe(df.head(5))
    
    st.subheader("DataFrame Info")
    st.write(f"**Columns:** {list(df.columns)}")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    null_count = df.isnull().sum().sum()
    st.write(f"**There are {null_count} null values in the DataFrame.**")
    
    if null_count > 0:
        option = st.selectbox(
            "How do you want to deal with null values?",
            ("Drop rows", "Drop columns", "Replace with 0", "Fill with mean", "Leave as is")
        )
        st.write(f"You selected: {option}")

        if option == "Drop rows":
            df = df.dropna()
            st.success("Rows with null values dropped.")
        elif option == "Drop columns":
            df = df.dropna(axis=1)
            st.success("Columns with null values dropped.")
        elif option == "Replace with 0":
            df = df.fillna(0)
            st.success("All null values replaced with 0.")
        elif option == "Fill with mean":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("All numeric null values filled with column mean.")
        else:
            st.info("Null values left as is.")

        st.subheader("DataFrame after null handling")
        st.dataframe(df.head(5))

    # --- Visualization Section ---
    st.subheader("Visualization")

    col1, col2 = st.columns(2)

    # Visualization 1 controls
    with col1:
        viz1 = st.selectbox(
            "Select first visualization:",
            ("Pairplot", "Correlation Heatmap", "Histogram"),
            key="viz1"
        )
        numeric_cols1 = df.select_dtypes(include=['number']).columns.tolist()
        selected_cols1, hue_col1, hist_cols1 = [], None, []
        if viz1 == "Pairplot" and len(numeric_cols1) >= 2:
            selected_cols1 = st.multiselect(
                "Select columns for pairplot (at least 2):",
                options=numeric_cols1,
                default=numeric_cols1[:2],
                key="pairplot_cols1"
            )
            hue_col1 = st.selectbox(
                "Select a column for hue (optional):",
                options=["None"] + list(df.columns),
                key="pairplot_hue1"
            )
        elif viz1 == "Histogram" and len(numeric_cols1) >= 1:
            hist_cols1 = st.multiselect(
                "Select columns for histogram (x-axis):",
                options=numeric_cols1,
                default=[numeric_cols1[0]] if numeric_cols1 else [],
                key="hist_cols1"
            )
        # No plot here, just controls

    # Visualization 2 controls
    with col2:
        viz2 = st.selectbox(
            "Select second visualization:",
            ("Pairplot", "Correlation Heatmap", "Histogram"),
            key="viz2"
        )
        numeric_cols2 = df.select_dtypes(include=['number']).columns.tolist()
        selected_cols2, hue_col2, hist_cols2 = [], None, []
        if viz2 == "Pairplot" and len(numeric_cols2) >= 2:
            selected_cols2 = st.multiselect(
                "Select columns for pairplot (at least 2):",
                options=numeric_cols2,
                default=numeric_cols2[:2],
                key="pairplot_cols2"
            )
            hue_col2 = st.selectbox(
                "Select a column for hue (optional):",
                options=["None"] + list(df.columns),
                key="pairplot_hue2"
            )
        elif viz2 == "Histogram" and len(numeric_cols2) >= 1:
            hist_cols2 = st.multiselect(
                "Select columns for histogram (x-axis):",
                options=numeric_cols2,
                default=[numeric_cols2[0]] if numeric_cols2 else [],
                key="hist_cols2"
            )
        # No plot here, just controls

    # Show both graphs together
    show_graphs = st.button("Show Graphs")

    if show_graphs:
        col1, col2 = st.columns(2)
        with col1:
            if viz1 == "Pairplot":
                if len(selected_cols1) >= 2:
                    plt.figure()
                    if hue_col1 and hue_col1 != "None":
                        pairplot_fig1 = sns.pairplot(df[selected_cols1 + [hue_col1]], hue=hue_col1)
                    else:
                        pairplot_fig1 = sns.pairplot(df[selected_cols1])
                    st.pyplot(pairplot_fig1)
                else:
                    st.info("Select at least 2 columns for pairplot.")
            elif viz1 == "Correlation Heatmap":
                if len(numeric_cols1) >= 2:
                    plt.figure(figsize=(8, 6))
                    corr1 = df[numeric_cols1].corr()
                    sns.heatmap(corr1, annot=True, cmap="coolwarm")
                    st.pyplot(plt.gcf())
                else:
                    st.info("Not enough numeric columns for heatmap.")
            elif viz1 == "Histogram":
                if len(hist_cols1) >= 1:
                    n = len(hist_cols1)
                    nrows = min(3, (n - 1) // 3 + 1)
                    ncols = min(3, n if n < 3 else 3)
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
                    axes = axes.flatten() if n > 1 else [axes]
                    for i, col in enumerate(hist_cols1):
                        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
                        axes[i].set_xlabel(col)
                        axes[i].set_title(f"Histogram of {col}")
                    # Hide any unused subplots
                    for j in range(i + 1, len(axes)):
                        axes[j].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Select at least one column for histogram.")

        with col2:
            if viz2 == "Pairplot":
                if len(selected_cols2) >= 2:
                    plt.figure()
                    if hue_col2 and hue_col2 != "None":
                        pairplot_fig2 = sns.pairplot(df[selected_cols2 + [hue_col2]], hue=hue_col2)
                    else:
                        pairplot_fig2 = sns.pairplot(df[selected_cols2])
                    st.pyplot(pairplot_fig2)
                else:
                    st.info("Select at least 2 columns for pairplot.")
            elif viz2 == "Correlation Heatmap":
                if len(numeric_cols2) >= 2:
                    plt.figure(figsize=(8, 6))
                    corr2 = df[numeric_cols2].corr()
                    sns.heatmap(corr2, annot=True, cmap="coolwarm")
                    st.pyplot(plt.gcf())
                else:
                    st.info("Not enough numeric columns for heatmap.")
            elif viz2 == "Histogram":
                if len(hist_cols2) >= 1:
                    n = len(hist_cols2)
                    nrows = min(3, (n - 1) // 3 + 1)
                    ncols = min(3, n if n < 3 else 3)
                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
                    axes = axes.flatten() if n > 1 else [axes]
                    for i, col in enumerate(hist_cols2):
                        sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
                        axes[i].set_xlabel(col)
                        axes[i].set_title(f"Histogram of {col}")
                    # Hide any unused subplots
                    for j in range(i + 1, len(axes)):
                        axes[j].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Select at least one column for histogram.")

    # --- Clustering Analysis Section ---
    st.subheader("Clustering Analysis")
    numeric_cols_for_clustering = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols_for_clustering:
        st.warning("No numeric columns available for clustering after preprocessing. Please check your data.")
    else:
        st.info("Ensure your data is scaled before clustering for best results.")
        clustering_algorithm = st.selectbox(
            "Select Clustering Algorithm:",
            ("K-Means",)  # You can add more algorithms later
        )

        if clustering_algorithm == "K-Means":
            st.subheader("K-Means Clustering")

            # Prompt user to drop variables before analysis
            drop_vars = st.multiselect(
                "Please select variables to drop before analysis",
                options=numeric_cols_for_clustering,
                key="drop_vars_for_clustering"
            )
            clustering_vars = [col for col in numeric_cols_for_clustering if col not in drop_vars]

            if len(clustering_vars) < 2:
                st.warning("Please keep at least 2 numeric variables for clustering analysis.")
            else:
                max_clusters = st.slider("Max K for Elbow Method:", min_value=2, max_value=10, value=5)

                # Elbow Method
                if st.button("Run Elbow Method"):
                    inertias = []
                    for k in range(1, max_clusters + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                        kmeans.fit(df[clustering_vars])
                        inertias.append(kmeans.inertia_)

                    # Standard Elbow Plot
                    fig, ax = plt.subplots()
                    ax.plot(range(1, max_clusters + 1), inertias, marker='o')
                    ax.set_title("Elbow Method for Optimal K")
                    ax.set_xlabel("Number of Clusters (K)")
                    ax.set_ylabel("Inertia")
                    st.pyplot(fig)
                    st.info("Look for the 'elbow' point where the decrease in inertia slows down significantly.")

                    # Yellowbrick KElbowVisualizer
                    st.subheader("KElbow Visualizer (Yellowbrick)")
                    fig2, ax2 = plt.subplots()
                    model = KMeans(random_state=42, n_init='auto')
                    visualizer = KElbowVisualizer(model, k=(2, max_clusters), ax=ax2)
                    visualizer.fit(df[clustering_vars])
                    visualizer.finalize()
                    st.pyplot(fig2)
                    st.info("The KElbow Visualizer helps you visually select the optimal number of clusters.")

                n_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=max_clusters, value=2)

                if st.button("Run K-Means Clustering"):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    df['cluster_label'] = kmeans.fit_predict(df[clustering_vars])
                    st.success(f"K-Means clustering performed with {n_clusters} clusters.")

                    # Visualize Clusters (PCA for 2D visualization)
                    st.subheader("Cluster Visualization (2D)")
                    if len(clustering_vars) >= 2:
                        pca = PCA(n_components=2)
                        principal_components = pca.fit_transform(df[clustering_vars])
                        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                        pca_df['cluster_label'] = df['cluster_label']

                        fig_pca = plt.figure(figsize=(10, 7))
                        sns.scatterplot(x='PC1', y='PC2', hue='cluster_label', data=pca_df, palette='viridis', legend='full')
                        plt.title('Clusters Visualized with PCA')
                        st.pyplot(fig_pca)

                        st.write("Explained variance ratio of PC1:", pca.explained_variance_ratio_[0])
                        st.write("Explained variance ratio of PC2:", pca.explained_variance_ratio_[1])
                    else:
                        st.warning("Not enough numeric columns to perform 2D PCA visualization. Need at least 2 numeric columns.")

                    # Cluster Profile Analysis
                    st.subheader("Cluster Profiles")
                    cluster_profiles = df.groupby('cluster_label')[clustering_vars].mean()
                    st.dataframe(cluster_profiles)
                    st.info("This table shows the average values of features for each cluster, helping to characterize them.")

                    # Silhouette Score
                    score = None
                    if n_clusters > 1:
                        try:
                            score = silhouette_score(df[clustering_vars], df['cluster_label'])
                            st.write(f"**Silhouette Score:** {score:.3f}")
                            st.info("A higher Silhouette Score (closer to 1) indicates better-defined clusters.")
                        except Exception as e:
                            st.error(f"Could not calculate Silhouette Score: {e}. This might happen if all points are in one cluster or other issues.")

                    # Store results in session_state for later summarization
                    st.session_state['cluster_profiles'] = cluster_profiles
                    st.session_state['score'] = score

    # --- Place this OUTSIDE the clustering button block ---
    if 'cluster_profiles' in st.session_state and 'score' in st.session_state and st.session_state['score'] is not None:
        summary_prompt = f"""
You are a data science assistant. Summarize the following cluster analysis results for a non-technical audience:

Cluster Profiles:
{st.session_state['cluster_profiles'].to_string()}

Silhouette Score: {st.session_state['score']:.3f}

Please provide a concise summary of what these clusters mean and any insights you can infer. Specifically, I want a summary of the clusters, their characteristics, and any notable patterns or insights that can be derived from the clustering analysis.
Make sure to explain the significance of the Silhouette Score and how it relates to the quality of the clustering.
"""
        @st.cache_data(show_spinner="Loading summary...", persist=True)
        def get_llm_summary(prompt):
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful data science assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.5,
            )
            return response.choices[0].message.content

        if st.button("Summarize Clustering Results with AI Agent"):
            with st.spinner("Summarizing with AI Agent..."):
                try:
                    summary = get_llm_summary(summary_prompt)
                    st.subheader("AI Agent Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Groq API error: {e}")
else:
    st.info("Please upload a CSV file to proceed.")