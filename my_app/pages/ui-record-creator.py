from ast import literal_eval

import argilla as rg
import spacy
import streamlit as st
import streamlit_analytics
from streamlit_tags import st_tags
from text_highlighter import text_highlighter
from utils.commons import (
    ArgillaSingleton,
    argilla_login_flow,
    get_data_snapshot,
    get_dataset_list,
)

st.set_page_config(
    page_title="Argilla - UI record creator",
    page_icon=":writing_hand::skin-tone-4:",
    layout="wide",
)

streamlit_analytics.start_tracking(load_from_json=f"{__file__}.json")

api_url, api_key = argilla_login_flow("UI record creator")

st.write(
    """
    This page allows you to create and annotate individual record from Argilla without using any code!
    In the background it uses `argilla.log()` and `TextClassificationRecord`, `TokenClassificationRecord`, and `Text2TextRecord`.
    """
)

nlp = spacy.blank("en")
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
    get_data_snapshot(dataset_argilla_name, dataset_argilla_workspace)
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


if dataset_argilla_name:
    dataset_type = st.selectbox("Dataset Type", options, disabled=disabled)
    if dataset_type in ["TextClassification", "TokenClassification"]:
        labels = st_tags(label="Labels", value=labels, text="Press enter to add more")

        if not any(labels):
            st.warning("No labels provided")
            st.stop()
        if dataset_type == "TextClassification":
            multi_label = st.radio("multi label", [False, True], horizontal=True)
        else:
            multi_label = False
    text = st.text_area("Text")

    if text:
        if dataset_type == "TextClassification":
            if multi_label:
                annotation = st.multiselect("annotation", labels, default=labels)
            else:
                annotation = st.radio("annotation", labels, horizontal=True)

            record = rg.TextClassificationRecord(
                text=text, annotation=annotation, multi_label=multi_label
            )
        elif dataset_type == "TokenClassification":
            annotation = text_highlighter(
                text=text,
                labels=labels,
            )
            if annotation:
                annotation = [(an["tag"], an["start"], an["end"]) for an in annotation]

            tokens = [token.text for token in nlp(text)]
            record = rg.TokenClassificationRecord(
                text=text, tokens=tokens, annotation=annotation
            )

        elif dataset_type == "Text2Text":
            annotation = st.text_area("Annotation")
            record = rg.Text2TextRecord(text=text, annotation=annotation)
        metadata = st.text_area("Metadata", value="{}")
        metadata = literal_eval(metadata)
        record.metadata = metadata
        st.write(record)
    else:
        st.warning("Please enter text")

    save = st.button("Save")
    if save:
        rg.log(record, dataset_argilla_name)
        st.success("Saved")
else:
    st.warning("Please enter dataset name")

streamlit_analytics.stop_tracking(save_to_json=f"{__file__}.json")
