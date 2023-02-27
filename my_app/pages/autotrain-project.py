
import argilla as rg
import streamlit as st
import streamlit_analytics
from huggingface_hub import ModelFilter
from utils.autotrain import get_projects, schedule_retrain, task_id_mapping
from utils.commons import (
    argilla_login_flow,
    get_data_snapshot,
    get_dataset_list,
    hf_login_flow,
)

st.set_page_config(
    page_title="Argilla - 🦾 - Autotrain Project",
    page_icon="🦾",
    layout="wide",
)

streamlit_analytics.start_tracking(load_from_json=f"{__file__}.json")

api_url, api_key = argilla_login_flow("🦾 Autotrain Project")

st.write(
    """
    This page allows you to train a model using [AutoTrain](https://ui.autotrain.huggingface.co) wihout using any code based on you Argilla datasets!
    In the background it uses `argilla.load().prepare_for_training()`, `datasets.push_to_hub()` and the [AutoTrain API](https://api.autotrain.huggingface.co/docs).
    """
)

hf_auth_token, api = hf_login_flow()

user_info = api.whoami()
namespaces = [user_info["name"]] + [org["name"] for org in user_info["orgs"]]

projects = get_projects(hf_auth_token)
project_ids = [proj["proj_name"] for proj in projects]

target_namespace = st.selectbox(
    "Hugging Face namespace",
    options=namespaces,
    help="the namespace where the trained model should end up",
)

datasets_list = [
    f"{ds['owner']}/{ds['name']}" for ds in get_dataset_list(api_url, api_key)
]
dataset_argilla = st.selectbox(
    "Argilla dataset name",
    options=datasets_list,
)
dataset_argilla_name = dataset_argilla.split("/")[-1]
dataset_argilla_workspace = dataset_argilla.split("/")[0]
get_data_snapshot(
    dataset_argilla_name, dataset_argilla_workspace, query="status: Validated"
)

input_model = st.text_input(
    "Input Model [from the hub](https://huggingface.co/models)",
    value="bert-base-uncased",
    help="the base model to re-train",
)

potential_models = api.list_models(filter=ModelFilter(model_name=input_model))
if not len(potential_models) == 1:
    if not any([(input_model == model.modelId) for model in list(potential_models)]):
        st.warning("Please select a model from the list below:")
        st.write(potential_models)
        st.stop()

for dataset in get_dataset_list(api_url, api_key):
    if (
        dataset["name"] == dataset_argilla_name
        and dataset["owner"] == dataset_argilla_workspace
    ):
        dataset_type = dataset["task"]
        break

if dataset_type == "TextClassification":
    task_options = ["text-classification-multi-class", "text-classification-binary"]
elif dataset_type == "TokenClassification":
    task_options = ["token-classification"]
elif dataset_type == "Text2Text":
    task_options = ["summarization"]

task = st.selectbox("Task", task_options)
task_id = task_id_mapping[task]

if task_id in [1, 2]:
    mapping = {
        "text": "text",
        "label": "target",
    }
elif task_id in [4]:
    mapping = {
        "tokens": "tokens",
        "ner_tags": "tags",
    }
elif task_id in [8]:
    mapping = {
        "text": "text",
        "target": "target",
    }


directly_train = False
free_training = st.checkbox("Train for free (max. 3000 samples)", value=True)
st.warning("Autotrain@HF is currently in beta and only allows public datasets, hence your data will published publically.")
start = st.button("Schedule Autotrain")

if start:
    with st.spinner(text="Export in progress..."):
        rg.set_workspace(dataset_argilla_workspace)
        if free_training:
            ds = rg.load(dataset_argilla_name, limit=3000)
        else:
            ds = rg.load(dataset_argilla_name)
        ds_ds = ds.prepare_for_training(framework="transformers", train_size=0.8)

        input_dataset = f"{target_namespace}/{dataset_argilla_name}"
        ds_ds.push_to_hub(
            input_dataset,
            token=hf_auth_token,
            private=False,
        )
        autotrain_project_name = (
            f"{dataset_argilla_name}-{api.dataset_info(input_dataset).sha[:7]}"
        )

    schedule_retrain(
        hf_auth_token=hf_auth_token,
        target_namespace=target_namespace,
        input_dataset=input_dataset,
        input_model=input_model,
        autotrain_project_prefix=autotrain_project_name,
        task_id=task_id,
        directly_train=directly_train,
        mapping=mapping,
    )


streamlit_analytics.stop_tracking(save_to_json=f"{__file__}.json")
