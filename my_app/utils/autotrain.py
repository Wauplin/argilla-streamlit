import time

import requests
import streamlit as st
from huggingface_hub.hf_api import HfApi
from pydantic import BaseModel

AUTOTRAIN_API_URL = "https://api.autotrain.huggingface.co"
AUTOTRAIN_UI_URL = "https://ui.autotrain.huggingface.co"

task_id_mapping = {
    "text-classification-binary": 1,
    "text-classification-multi-class": 2,
    "token-classification": 4,
    # "question-answering-extractive": 5,
    "summarization": 8,
    # "text-regression": 10,
    # "image-multi-class-classification": 18,
    # "tabular-data-regression": 16,
    # "tabular-data-multi-label": 15,
    # "tabular-data-multi-class": 14,
    # "tabular-data-binary": 13,
}


class AutoTrainInfo(BaseModel):
    hf_auth_token: str
    target_namespace: str
    input_dataset: str
    input_model: str
    autotrain_project_prefix: str
    task_id: int
    project_id: str = None
    directly_train: bool
    mapping: dict


def get_projects(hf_auth_token):
    return AutoTrain.list_projects(hf_auth_token)


def schedule_retrain(
    hf_auth_token,
    target_namespace,
    input_dataset,
    input_model,
    autotrain_project_prefix,
    task_id,
    directly_train,
    mapping,
):
    payload = AutoTrainInfo(
        hf_auth_token=hf_auth_token,
        target_namespace=target_namespace,
        input_dataset=input_dataset,
        input_model=input_model,
        autotrain_project_prefix=autotrain_project_prefix,
        task_id=task_id,
        directly_train=directly_train,
        mapping=mapping,
    )
    # Create the autotrain project
    try:
        project = AutoTrain.create_project(payload)
        payload.project_id = project["id"]
        AutoTrain.add_data(payload)
        AutoTrain.start_processing(payload)
        if payload.directly_train:
            AutoTrain.start_training(payload)
    except requests.HTTPError as err:
        print("ERROR while requesting AutoTrain API:")
        print(f"  code: {err.response.status_code}")
        print(f"  {err.response.json()}")
        raise
    # Notify in the community tab
    notify_success(payload)

    return {"processed": True}


class AutoTrain:
    @staticmethod
    def list_projects(hf_auth_token) -> str:
        projects = requests.get(
            f"{AUTOTRAIN_API_URL}/projects/list",
            headers={"Authorization": f"Bearer {hf_auth_token}"},
        )
        projects.raise_for_status()
        return projects.json()

    @staticmethod
    def create_project(payload: AutoTrainInfo) -> dict:
        project_resp = requests.post(
            f"{AUTOTRAIN_API_URL}/projects/create",
            json={
                "username": payload.target_namespace,
                "proj_name": payload.autotrain_project_prefix,
                "task": payload.task_id,
                "config": {
                    "hub-model": payload.input_model,
                    "max_models": 1,
                    "language": "en",
                },
            },
            headers={"Authorization": f"Bearer {payload.hf_auth_token}"},
        )

        project_resp.raise_for_status()
        return project_resp.json()

    @staticmethod
    def add_data(payload: AutoTrainInfo):
        requests.post(
            f"{AUTOTRAIN_API_URL}/projects/{payload.project_id}/data/dataset",
            json={
                "dataset_id": payload.input_dataset,
                "dataset_split": "train",
                "split": 1,
                "col_mapping": payload.mapping,
            },
            headers={
                "Authorization": f"Bearer {payload.hf_auth_token}",
            },
        ).raise_for_status()
        requests.post(
            f"{AUTOTRAIN_API_URL}/projects/{payload.project_id}/data/dataset",
            json={
                "dataset_id": payload.input_dataset,
                "dataset_split": "test",
                "split": 2,
                "col_mapping": payload.mapping,
            },
            headers={
                "Authorization": f"Bearer {payload.hf_auth_token}",
            },
        ).raise_for_status()

    @staticmethod
    def start_processing(payload: AutoTrainInfo):
        resp = requests.post(
            f"{AUTOTRAIN_API_URL}/projects/{payload.project_id}/data/start_processing",
            headers={
                "Authorization": f"Bearer {payload.hf_auth_token}",
            },
        )
        resp.raise_for_status()
        return resp

    @staticmethod
    def start_training(payload: AutoTrainInfo):
        succeeded = False
        with st.spinner("Waiting for data to be processed..."):
            while not succeeded:
                resp = requests.post(
                    f"{AUTOTRAIN_API_URL}/projects/{payload.project_id}/data/start_processing",
                    headers={
                        "Authorization": f"Bearer {payload.hf_auth_token}",
                    },
                )
                time.sleep()
                if resp.status_code == 200:
                    succeeded = True

        return resp


def notify_success(payload: AutoTrainInfo):
    message = NOTIFICATION_TEMPLATE.format(
        input_model=payload.input_model,
        input_dataset=payload.input_dataset,
        project_id=payload.project_id,
        ui_url=AUTOTRAIN_UI_URL,
    )
    st.success(message)
    return HfApi(token=payload.hf_auth_token).create_discussion(
        repo_id=payload.input_dataset,
        repo_type="dataset",
        title="✨ Retraining started!",
        description=message,
        token=payload.hf_auth_token,
    )


NOTIFICATION_TEMPLATE = """\
🌸 Hello there!
Following an update of [{input_dataset}](https://huggingface.co/datasets/{input_dataset}), an automatic training of [{input_model}](https://huggingface.co/{input_model}) has been scheduled on AutoTrain!
Please review and approve the project [here]({ui_url}/{project_id}/trainings) to start the training job.
(This is an automated message)
"""
