# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import streamlit_analytics

st.set_page_config(page_title="Argilla Streamlit", page_icon="👋", layout="wide")

streamlit_analytics.start_tracking(load_from_json=f"{__file__}.json")

st.image("https://docs.argilla.io/en/latest/_static/images/logo-light-mode.svg")

st.write("# Welcome to Argilla Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.success(
    "PRs are welcome! 🙌 [Github repo](https://github.com/argilla-io/argilla-streamlit)"
)
st.markdown(
    """
    Argilla is a production-ready framework for building and improving datasets for NLP projects. This repo is focused on extended UI functionalities for Argilla. 👑

    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!

    ## Next Steps
    If you want to continue learning Argilla:
    - 🙋‍♀️ Join the [Argilla Slack Community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g)
    - ⭐ Argilla [Github repo](https://github.com/argilla-io/argilla)
    - 📚 Argilla [documentation](https://docs.argilla.io) for more guides and tutorials.
    """
)

streamlit_analytics.stop_tracking(save_to_json=f"{__file__}.json")
