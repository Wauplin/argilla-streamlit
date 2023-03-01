# import os

# import argilla as rg
# import streamlit as st
# import streamlit_analytics
# from _utils import login_workflow
# from text_highlighter import text_highlighter

# st.set_page_config(
#     page_title="Argilla Annotation Guideline and Comment Ability",
#     page_icon=":memo:",
#     layout="wide",
# )

# # st.image("https://docs.argilla.io/en/latest/_static/images/logo-light-mode.svg")
# st.title("Annotation Comment and Note support")

# # login workflow
# login_workflow()

# st.error(
#     "WIP: Work in progress. Check our https://github.com/argilla-io/argilla-streamlit"
#     " to open a PR."
# )
# st.stop()
# dataset = st.text_input("Dataset Name")

# if dataset:
#     records = rg.load(name=dataset, limit=1)

#     if records:
#         record = records[0]
#         if isinstance(record, rg.TokenClassificationRecord) or isinstance(
#             record, rg.TextClassificationRecord
#         ):
#             labels = st.text_input("Labels")
#             split_labels = labels.split(",")
#             split_labels = [label.strip() for label in split_labels]

#             if not any(split_labels):
#                 st.warning("No labels provided")
#                 st.stop()
#             if isinstance(record, rg.TokenClassificationRecord):
#                 multi_label = st.radio("multi label", [False, True], horizontal=True)
#             else:
#                 multi_label = False
# else:
#     st.warning("No dataset provided")
#     st.stop()

# st.write("This is an annotation guideline. Label A is for cats, label B is for dogs.")
# query = st.text_input("Query", value="status: Default", key="query")
# if not query:
#     query = None

# records = rg.load(name=dataset, limit=1, query=query)


# def form_callback(dataset, query):
#     rg.log(st.session_state.rec, dataset)
#     st.session_state.rec = rg.load(name=dataset, limit=1, query=query)[0]
#     if st.session_state.rec.inputs is not None:
#         st.session_state.inputs = "\n".join(
#             [
#                 f"**{key}** \n\n {value}"
#                 for key, value in st.session_state.rec.inputs.items()
#             ]
#         )
#     else:
#         st.session_state.inputs = st.session_state.rec.text
#     st.session_state.comment = st.session_state.rec.metadata.get("comment", "")
#     if st.session_state.rec.annotation:
#         st.session_state["annotation"] = st.session_state.rec.annotation

#     st.success("Saved")


# if records:
#     with st.form(key="my_form"):
#         records = records[0]
#         st.session_state.rec = records
#         if isinstance(st.session_state.rec, rg.TokenClassificationRecord):
#             if st.session_state.rec.annotation:
#                 old_annotation = [
#                     {
#                         "start": an[1],
#                         "end": an[2],
#                         "tag": an[0],
#                         "text": st.session_state.rec.text[an[1] : an[2]],
#                     }
#                     for an in st.session_state.rec.annotation
#                 ]
#             else:
#                 old_annotation = None
#             annotation = text_highlighter(
#                 text=st.session_state.rec.text,
#                 labels=split_labels,
#                 annotations=old_annotation,
#             )
#             formatted_annotation = [
#                 (an["tag"], an["start"], an["end"]) for an in annotation
#             ]

#         elif isinstance(st.session_state.rec, rg.TextClassificationRecord):
#             if st.session_state.rec.inputs is not None:
#                 st.text_area(
#                     "Text",
#                     value="\n".join(
#                         [
#                             f"{key}: {value}"
#                             for key, value in st.session_state.rec.inputs.items()
#                         ]
#                     ),
#                     key="inputs",
#                     disabled=True,
#                 )
#             else:
#                 st.text_area(
#                     "Text", value=st.session_state.rec.text, key="inputs", disabled=True
#                 )

#             if st.session_state.rec.multi_label:
#                 annotation = st.multiselect(
#                     "annotation",
#                     split_labels,
#                     st.session_state.rec.annotation,
#                     key="annotation",
#                 )
#             else:
#                 if st.session_state.rec.annotation:
#                     if st.session_state.rec.annotation in split_labels:
#                         index = split_labels.index(st.session_state.rec.annotation)
#                     else:
#                         st.error(st.session_state.rec.annotation + " not in labels")
#                 else:
#                     index = 0
#                 annotation = st.radio(
#                     "annotation",
#                     split_labels,
#                     index,
#                     horizontal=True,
#                     key="annotation",
#                 )

#         elif isinstance(st.session_state.rec, rg.Text2TextRecord):
#             st.write(st.session_state.rec.text)
#             st.text_area(st.session_state.rec.annotation)

#         try:
#             st.session_state.rec.__class__(**st.session_state.rec.__dict__)
#             st.session_state.rec.annotation = annotation
#         except Exception as e:
#             st.write(e)

#         if st.session_state.rec.metadata:
#             if "comment" in st.session_state.rec.metadata:
#                 input_comment = st.session_state.rec.metadata["comment"]
#             else:
#                 input_comment = ""
#         else:
#             input_comment = ""

#         comment = st.text_input("comment", value=input_comment, key="comment")
#         if st.session_state.rec.metadata:
#             st.session_state.rec.metadata["comment/note"] = comment
#         else:
#             st.session_state.rec.metadata = {"comment": comment}

#         save = st.form_submit_button(
#             "Save", on_click=form_callback, args=(dataset, query)
#         )

# else:
#     st.warning("No records found")


# 