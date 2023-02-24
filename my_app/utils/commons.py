import os

import argilla as rg
import httpx
import streamlit as st
from argilla.client.api import ArgillaSingleton, active_client
from argilla.client.apis.datasets import __TASK_TO_SETTINGS__
from huggingface_hub import HfApi


def argilla_login_flow(title: str) -> str:
    """
    It tries to log in to Argilla
    using the environment variables `ARGILLA_API_URL` and `ARGILLA_API_KEY`. If they
    are not set, it shows a sidebar with two text inputs to manually enter the API URL
    and API Key

    Args:
      title: The title of the app.

    Returns:
      The api_url is being returned.
    """
    st.image("https://docs.argilla.io/en/latest/_static/images/logo-light-mode.svg")

    api_url = None
    if os.environ.get("ARGILLA_API_URL") and os.environ.get("ARGILLA_API_KEY"):
        api_url = os.environ.get("ARGILLA_API_URL")
        api_key = os.environ.get("ARGILLA_API_KEY")
        rg.init(
            api_url=api_url,
            api_key=api_key,
        )
        st.success(
            f"Logged in at {os.environ.get('ARGILLA_API_URL')}, and workspace is"
            f" {rg.get_workspace()}. Change `ARGILLA_API_URL` and `ARGILLA_API_KEY` to"
            " use a different endpoint."
        )
    else:
        try:
            api_url = st.sidebar.text_input(
                "API URL", "https://argilla-live-demo.hf.space"
            )
            api_key = st.sidebar.text_input("API Key", value="team.apikey")
            rg.init(
                api_url=api_url,
                api_key=api_key,
            )

            st.success(
                f"Logged in at {api_url}, and workspace is {rg.get_workspace()}. Set"
                " `ARGILLA_API_URL` and `ARGILLA_API_KEY` as environment variables to"
                " avoid this step."
            )
        except Exception:
            st.error(
                "Invalid API URL or API Key. Use a correct manual input or, even"
                " better, set `ARGILLA_API_URL` and `ARGILLA_API_KEY` as environment"
                " variables to avoid this step."
            )
    st.title(title)
    return api_url, api_key


def get_data_snapshot(dataset_name, workspace, query=None):
    rg.set_workspace(workspace)
    if query == "":
        query = None
    ds = rg.load(dataset_name, query=query, limit=5).to_pandas()
    st.write(f"Sample of the `{workspace}/{dataset_name}`", ds)


def hf_login_flow():
    """
    It checks if the user has provided a Hugging Face API token in the environment variable
    `HF_AUTH_TOKEN` or in the sidebar. If not, it displays an error message and stops the app

    Returns:
      A tuple of the token and the api
    """
    hf_auth_token = os.environ.get("HF_AUTH_TOKEN", "")
    if not hf_auth_token:
        hf_auth_token = st.sidebar.text_input(
            "Hugging Face [User Access Tokens](https://huggingface.co/settings/tokens)",
            os.environ.get("HF_AUTH_TOKEN", ""),
        )
    if not hf_auth_token:
        st.error(
            "Please provide a Hugging Face [User Access"
            " Tokens](https://huggingface.co/settings/tokens) in the sidebar or set"
            " `HF_AUTH_TOKEN` as environment variable"
        )
        st.stop()
    api = HfApi(token=hf_auth_token)
    return hf_auth_token, api


# def record_info():
#     with st.expander("Dataset Type Info"):
#         if dataset_type == "TextClassification":
#             st.write(rg.TextClassificationRecord.__doc__)
#         elif dataset_type == "TokenClassification":
#             st.write(rg.TokenClassificationRecord.__doc__)
#         else:
#             st.write(rg.Text2TextRecord.__doc__)


def get_dataset_list(api_url, api_key):
    client = ArgillaSingleton.init(api_url, api_key)._client
    try:
        url = "{}/api/datasets".format(client.base_url)

        response = httpx.get(
            url=url,
            headers=client.get_headers(),
            cookies=client.get_cookies(),
            timeout=client.get_timeout(),
        )
        response.raise_for_status()
        response.json()
    except Exception:
        url = "{}/api/datasets/".format(client.base_url)

        response = httpx.get(
            url=url,
            headers=client.get_headers(),
            cookies=client.get_cookies(),
            timeout=client.get_timeout(),
        )
        response.raise_for_status()

    dataset_overview = []
    for dataset in response.json():
        metadata = dataset["metadata"].values()
        if (
            metadata
            and not isinstance(list(metadata)[0], str)
            and not isinstance(list(metadata)[0], int)
        ):
            metadata = list(metadata)[0].get(
                "labels", list(metadata)[0].get("entities")
            )
        else:
            metadata = None
        dataset_overview.append(
            {
                "name": dataset["name"],
                "task": dataset["task"],
                "owner": dataset["owner"],
                "id": dataset["id"],
                "labels": metadata,
            }
        )
        # if metadata is None:
        #     # dataset_overview[-1]["labels"] =
        #     setting = get_dataset_settings(dataset["name"], dataset["task"])
        #     if setting is not None:
        #         dataset_overview[-1]["labels"] = list(setting)
    return dataset_overview


def whoami():
    client = active_client()._client
    url = "{}/api/me".format(client.base_url)
    response = httpx.get(
        url=url,
        headers=client.get_headers(),
        cookies=client.get_cookies(),
        timeout=client.get_timeout(),
    )
    return {**response.json()}


def get_dataset_settings(dataset_name, dataset_task):
    client = active_client()._client
    url = f"{client.base_url}/api/datasets/{dataset_task}/{dataset_name}/settings"
    response = httpx.get(
        url=url,
        headers=client.get_headers(),
        cookies=client.get_cookies(),
        timeout=client.get_timeout(),
    )

    try:
        response.raise_for_status()
        return (
            __TASK_TO_SETTINGS__.get(dataset_task)
            .from_dict(response.json())
            .label_schema
        )
    except Exception:
        return None
