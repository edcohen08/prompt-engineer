import streamlit as st

st.set_page_config(page_title="AI Prompt Engineer", page_icon="🦫")

if 'prompt_count' not in st.session_state:
    st.session_state.prompt_count = 0

if 'prompt_0' not in st.session_state:
    st.session_state['prompt_0'] = ""

st.header("AI Prompt Engineer")
col_1, col_2 = st.columns(2)

for i in range(0, st.session_state.prompt_count + 1):
    with col_1:
        st.text_input(f"Prompt Candidate {i + 1}", help="no help", key=f"prompt_{i}", placeholder=f"Prompt candidate {i+1}", label_visibility="collapsed")
    with col_2:
        st.button('Remove prompt', key=f"remove_prompt_{i}")

if st.button('Add prompt', type="primary"):
    st.session_state.prompt_count += 1
    st.session_state[f"prompt_{st.session_state.prompt_count}"] = ""
    st.experimental_rerun()
