import streamlit as st

from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI, PromptLayerOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from pandas import DataFrame

@st.cache_resource
def load_evaluator_chain():
    template = """Give a score 0-100 to the AI output for how exactly it matches the ground truth.
    Do not penalize exact matches. Ground truth: {answer} AI: {ai_answer}"""
    prompt = PromptTemplate(
        input_variables=["answer", "ai_answer"],
        template=template,
    )
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return LLMChain(llm=llm, prompt=prompt, output_key="score")

@st.cache_resource
def load_zero_shot_chain():
    template = """{prompt_candidate}. Q: {input} A:"""

    prompt = PromptTemplate(
        input_variables=["prompt_candidate", "input"],
        template=template
    )
    llm = PromptLayerOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return LLMChain(llm=llm, prompt=prompt, output_key="ai_answer")

@st.cache_resource
def load_zero_shot_pipeline():
    zero_shot = load_zero_shot_chain()
    evaluator = load_evaluator_chain()
    return SequentialChain(
        chains=[zero_shot, evaluator],
        input_variables=["prompt_candidate", "input", "answer"],
        output_variables=["ai_answer", "score"]
    )

@st.cache_data
def call_zero_shot_pipeline(state: dict) -> DataFrame:
    zero_shot_pipeline = load_zero_shot_pipeline()
    results = []
    for i in range(st.session_state["prompt_count"] + 1):
        for j in range(st.session_state["demonstration_count"] + 1):
            results.append(zero_shot_pipeline({"prompt_candidate": st.session_state[f"prompt_{i}"], "input": st.session_state[f"question_{j}"], "answer": st.session_state[f"answer_{j}"]}))
    
    return DataFrame(results) 

@st.cache_data
def convert_df(df: DataFrame) -> str:
    return df.to_csv().encode("utf-8")

@st.cache_resource
def load_few_shot_chain():
    pass


