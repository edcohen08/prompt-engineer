import streamlit as st

st.set_page_config(page_title="AI Prompt Engineer", page_icon="🦫", layout="wide")

from models import call_zero_shot_pipeline, convert_df  # noqa E402

df = None

if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0

if "prompt_0" not in st.session_state:
    st.session_state["prompt_0"] = ""

if "demonstration_count" not in st.session_state:
    st.session_state.demonstration_count = 0

if "question_0" not in st.session_state:
    st.session_state["question_0"] = ""

if "answer_0" not in st.session_state:
    st.session_state["answer_0"] = ""

st.header("AI Prompt Engineer")
st.sidebar.header("API Keys")
with st.sidebar:
    st.text_input(
        "OpenAI API Key",
        key="openai_api_key",
        placeholder="OpenAI API Key*",
        type="password",
        label_visibility="collapsed",
    )
with st.sidebar:
    st.text_input(
        "Promptlayer API Key",
        key="promptlayer_api_key",
        placeholder="Promptlayer API Key",
        type="password",
        label_visibility="collapsed",
    )
col_1, col_2, col_3, col_4 = st.columns(4, gap="medium")

with st.form("prompt_test"):

    for i in range(0, st.session_state.prompt_count + 1):
        with col_1:
            st.text_area(
                f"Prompt Candidate {i+1} Template name",
                key=f"prompt_name_{i}",
                placeholder=f"Prompt candidate Template {i+1} name",
                label_visibility="collapsed",
            )
        with col_2:
            st.text_area(
                f"Prompt Candidate {i + 1}",
                help="no help",
                key=f"prompt_{i}",
                placeholder=f"Prompt candidate {i+1}",
                label_visibility="collapsed",
            )
    for i in range(0, st.session_state.demonstration_count + 1):
        with col_3:
            st.text_area(
                f"Question {i + 1}",
                help="no help",
                key=f"question_{i}",
                placeholder=f"question {i+1}",
                label_visibility="collapsed",
            )
        with col_4:
            st.text_area(
                f"Answer {i + 1}",
                help="no help",
                key=f"answer_{i}",
                placeholder=f"answer {i+1}",
                label_visibility="collapsed",
            )

    if st.form_submit_button():
        df = call_zero_shot_pipeline(dict(st.session_state))
        st.dataframe(df)

if df is not None:
    st.download_button(
        label="Download data as CSV",
        data=convert_df(df),
        file_name="results.csv",
        mime="text/csv",
    )

with col_1:
    if st.button("Add", type="primary", key="add_prompt"):
        st.session_state.prompt_count += 1
        st.session_state[f"prompt_{st.session_state.prompt_count}"] = ""
        st.experimental_rerun()

with col_3:
    if st.button("Add", type="primary", key="add_demonstration"):
        st.session_state.demonstration_count += 1
        st.session_state[f"question_{st.session_state.demonstration_count}"] = ""
        st.experimental_rerun()
