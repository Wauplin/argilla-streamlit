import argilla as rg
import numpy as np
import plotly.express as px
import spacy
import streamlit as st
import streamlit_analytics
from sklearn.decomposition import PCA
from streamlit_plotly_events import plotly_events
from streamlit_tags import st_tags
from umap import UMAP
from utils.commons import argilla_login_flow, get_data_snapshot, get_dataset_list

st.set_page_config(
    page_title="Argilla - 🏹 - Vector Annotator",
    page_icon="🏹",
    layout="wide",
)

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# streamlit_analytics.start_tracking(load_from_json=f"{__file__}.json")

# login workflow
api_url, api_key = argilla_login_flow("🏹 Vector Annotator")

st.write(
    """
    This page allows you to annotate bulks of records from Argilla based on their [semantic vectors](https://docs.argilla.io/en/latest/guides/label_records_with_semanticsearch.html) without using any code!
    Select a subset of the data using lasso select and get labelling!
    In the background it uses `argilla.load()`, `umap-learn`, `pca`, `pandas`, and `spacy`.
    """
)

datasets_list = [
    f"{ds['owner']}/{ds['name']}" for ds in get_dataset_list(api_url, api_key)
]
dataset_argilla = st.selectbox("Argilla Dataset Name", options=datasets_list)
dataset_argilla_name = dataset_argilla.split("/")[-1]
dataset_argilla_workspace = dataset_argilla.split("/")[0]
get_data_snapshot(dataset_argilla_name, dataset_argilla_workspace)
rg.set_workspace(dataset_argilla_workspace)
labels = []

for dataset in get_dataset_list(api_url, api_key):
    if (
        dataset["name"] == dataset_argilla_name
        and dataset["owner"] == dataset_argilla_workspace
    ):
        labels = dataset["labels"]
        if dataset["task"] != "TextClassification":
            st.warning("Only TextClassificationRecord is supported")
            st.stop()
labels = st_tags(label="Labels", value=labels, text="Press enter to add more")

st.info(
    "Information is cached but it is recommended to use a subset of the data through"
    " setting a maximum number of records or by specifying the selection through"
    " querying."
)
fast = st.checkbox("Fast mode (PCA) or accurate mode (UMAP)", value=True)
n_records = st.number_input(
    "Max number of records to query and analyze",
    min_value=1000,
    max_value=10000,
    value=2000,
)
query = st.text_input(
    "Query to filter records (optional). See [query"
    " syntax](https://docs.argilla.io/en/latest/guides/query_datasets.html)",
)

if dataset_argilla_name and labels:

    @st.cache(allow_output_mutation=True)
    def load_dataset(dataset_argilla_name, query, fast_mode, limit):
        if query and query is not None:
            query = f"({query}) AND vectors: *"
        else:
            query = "vectors: *"

        ds = rg.load(dataset_argilla_name, query=query, limit=limit)
        df = ds.to_pandas()

        if df.empty:
            st.warning("No dataset found")
            st.stop()
        vectors = df.vectors.values
        if len(list(vectors[0].keys())) > 1:
            vector_name = st.selectbox("Select vector", list(vectors[0].keys()))
        else:
            vector_name = list(vectors[0].keys())[0]
        vectors = np.array([v[vector_name] for v in vectors])

        if fast_mode:
            # Reduce the dimensions with UMAP
            umap = UMAP()
            X_tfm = umap.fit_transform(vectors)
        else:
            pca = PCA(n_components=2)
            X_tfm = pca.fit_transform(vectors)

        # # Apply coordinates
        df["x"] = X_tfm[:, 0]
        df["y"] = X_tfm[:, 1]

        sentencized_docs = nlp.pipe(df.text.values)
        sentencized_text = [
            "<br>".join([sent.text for sent in doc.sents]) for doc in sentencized_docs
        ]
        df["formatted_text"] = sentencized_text
        return df, ds

    df, ds = load_dataset(dataset_argilla_name, query, fast, n_records)
    fig = px.scatter(
        data_frame=df,
        x="x",
        y="y",
        hover_data={
            "formatted_text": True,
            "prediction": True,
            "annotation": True,
            "x": False,
            "y": False,
        },
        title="Records to Highlight",
    )

    selected_points = plotly_events(fig, select_event=True, click_event=False)
    point_index = [point["pointIndex"] for point in selected_points]

    if point_index:
        # filter dataframe based on selected points
        df_new = df.copy(deep=True)
        df_new = df_new.iloc[point_index]
        st.write(f"{len(df_new)} Selected Records")
        st.dataframe(df_new[["text", "prediction", "annotation"]])
        if ds[0].multi_label:
            annotation = st.multiselect("annotation", labels, default=labels)
        else:
            annotation = st.radio("annotation", labels, horizontal=True)
        del df_new["formatted_text"]

        if st.button("Annotate"):
            df_new["annotation"] = annotation
            ds_update = rg.read_pandas(df_new, task="TextClassification")
            rg.log(ds_update, api_url=api_url)
    else:
        st.warning("No point selected")
else:
    st.warning("Please enter a dataset name and labels")

# streamlit_analytics.stop_tracking(save_to_json=f"{__file__}.json")
