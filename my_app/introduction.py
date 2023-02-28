# Contents of ~/my_app/streamlit_app.py
import streamlit as st

st.set_page_config(page_title="Argilla Streamlit", page_icon="👋", layout="wide")

# streamlit_analytics.start_tracking(load_from_json=f"{__file__}.json")

x = st.columns(3)
x[0].image("https://docs.argilla.io/en/latest/_static/images/logo-light-mode.svg", use_column_width=True)

st.write("# Welcome to Argilla Streamlit! 👋")

st.sidebar.success("Select on of the apps above.")

st.success(
    "PRs are welcome! 🙌 [Github repo](https://github.com/argilla-io/argilla-streamlit)"
)
st.markdown(
    """
    Argilla is a production-ready framework for building and improving datasets for NLP projects. This repo is focused on extended UI functionalities for Argilla. 👑

    **👈 Select an app from the sidebar** to see some examples
    of what Argilla Streamlit Customs can do!

    ## Next Steps
    If you want to continue learning Argilla:
    - 🙋‍♀️ Join the [Argilla Slack Community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g)
    - ⭐ Argilla [Github repo](https://github.com/argilla-io/argilla)
    - 📚 Argilla [documentation](https://docs.argilla.io) for more guides and tutorials.
    """
)


# streamlit_analytics.stop_tracking(save_to_json=f"{__file__}.json")
