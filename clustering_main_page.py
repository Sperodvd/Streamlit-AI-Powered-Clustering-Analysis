import streamlit as st


about_page = st.Page(
    page="pages/Title_page.py",
    title="About",
    icon=":material/info:",
    default=True,
)

project_page = st.Page(
    page="pages/clustering_project.py",
    title="Clustering Analysis",
    icon=":material/analytics:",
)

pg = st.navigation(pages=[about_page, project_page])
pg.run()
