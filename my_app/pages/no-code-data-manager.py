from io import BytesIO

import argilla as rg
import pandas as pd
import spacy
import streamlit as st
import streamlit_analytics
import xlsxwriter
from utils.commons import ArgillaSingleton, argilla_login_flow, get_dataset_list

st.set_page_config(
    page_title="Argilla NoCode Data Manager", page_icon="💾", layout="wide"
)

# streamlit_analytics.start_tracking(load_from_json=f"{__file__}.json")

api_url, api_key = argilla_login_flow("No-code data manager")

st.write(
    """
    This page allows you to upload and download datasets from Argilla without using any code!
    In the background it uses `argilla.log()` and `pandas`. This requires you to have a valid `.csv`, `.xlsx` or `.xlsx`.
    """
)

action = st.sidebar.selectbox("Action", ["✍️ Upload Dataset", "💾 Download dataset"])

if action == "✍️ Upload Dataset":
    st.subheader(action)
    datasets_list = [
        f"{ds['owner']}/{ds['name']}" for ds in get_dataset_list(api_url, api_key)
    ]
    dataset_argilla = st.selectbox(
        "Argilla Dataset Name", options=["other"] + datasets_list
    )
    if dataset_argilla == "other":
        ArgillaSingleton.init(api_url, api_key)
        dataset_argilla_name = st.text_input("New Dataset Name")
        labels = []
        disabled = False
        options = ["TextClassification", "TokenClassification", "Text2Text"]
    else:
        dataset_argilla_name = dataset_argilla.split("/")[-1]
        dataset_argilla_workspace = dataset_argilla.split("/")[0]
        rg.set_workspace(dataset_argilla_workspace)
        for dataset in get_dataset_list(api_url, api_key):
            if (
                dataset["name"] == dataset_argilla_name
                and dataset["owner"] == dataset_argilla_workspace
            ):
                labels = dataset["labels"]
                dataset_type = dataset["task"]
                disabled = True
                options = [dataset_type]
                break

    dataset_type = st.selectbox("Dataset Type", options, disabled=disabled)

    if dataset_argilla_name is not None and dataset_argilla_name.strip() != "":
        records = []
        uploaded_file = st.file_uploader(
            "Upload your CSV or XLSX/XLS file", type=["csv", "xls", "xlsx"]
        )

        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file, sheet_name=0)
            except Exception:
                df = pd.read_csv(uploaded_file)

            st.write("Dataset preview:", df.head())
            string_columns = [col for col in df.columns if df[col].dtype == "object"]
            if len(string_columns) > 0:
                if dataset_type == "TextClassification":
                    column_select = st.multiselect("Select columns", string_columns)
                    if column_select:
                        records = []
                        for i, row in df[column_select].iterrows():
                            record = rg.TextClassificationRecord(
                                inputs={col: row[col] for col in column_select}
                            )
                            records.append(record)
                elif dataset_type == "TokenClassification":
                    column_select = st.selectbox("Select a Column", string_columns)
                    if column_select:
                        # Load the spaCy en_core_web_sm model
                        nlp = spacy.blank("en")
                        # Create a new column in the DataFrame with the tokenized text
                        df["tokenized_text"] = df[column_select].apply(
                            lambda x: [token.text for token in nlp(x)]
                        )
                        st.write("Tokenized Text:", df["tokenized_text"].head(3))
                        for i, row in df.iterrows():
                            record = rg.TokenClassificationRecord(
                                text=row[column_select],
                                tokens=row["tokenized_text"],
                            )
                            records.append(record)
                else:
                    column_select = st.selectbox("Select a Column", string_columns)
                    if column_select:
                        records = []
                        for i, row in df[column_select].iterrows():
                            record = rg.Text2TextRecord(text=row[column_select])
                            records.append(record)
                if len(records) > 0:
                    if st.button("Log data into Argilla"):
                        output = rg.log(records=records, name=dataset_argilla_name)
                        st.write(output)
                        st.write(f"{output.processed} records added to {api_url}")
    else:
        st.warning("Please provide a dataset name")

elif action == "💾 Download dataset":
    st.subheader(action)
    datasets_list = [
        f"{ds['owner']}/{ds['name']}" for ds in get_dataset_list(api_url, api_key)
    ]
    dataset_argilla = st.selectbox("Argilla Dataset Name", options=datasets_list)
    dataset_argilla_name = dataset_argilla.split("/")[-1]
    dataset_argilla_workspace = dataset_argilla.split("/")[0]
    query = st.text_input(
        "Query to filter records (optional). See [query"
        " syntax](https://docs.argilla.io/en/latest/guides/query_datasets.html)"
    )
    search = st.button("Search")
    if search:
        rg.set_workspace(dataset_argilla_workspace)
        dataset = rg.load(dataset_argilla_name, query=query).to_pandas()
        st.write("Dataset preview:", dataset.head())
        cols = st.columns(3)
        cols[0].download_button(
            label="Download as CSV",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=f"{dataset_argilla_name}.csv",
            mime="text/csv",
        )
        output = BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            dataset.to_excel(writer, sheet_name="Sheet1")
            writer.save()
            cols[1].download_button(
                label="Download as Excel",
                data=output,
                file_name=f"{dataset_argilla_name}.xlsx",
                mime="application/vnd.ms-excel",
            )
        cols[2].download_button(
            label="Download as JSON!",
            data=dataset.to_json(orient="records", lines=True).encode("utf-8"),
            file_name=f"{dataset_argilla_name}.json",
            mime="application/json",
        )
    else:
        st.info("Press the search button to load the dataset with the given query")


# streamlit_analytics.stop_tracking(save_to_json=f"{__file__}.json")
