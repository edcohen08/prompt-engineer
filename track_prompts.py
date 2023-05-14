from re import match
from pandas import DataFrame
from promptlayer import track


def write_to_prompt_layer(df: DataFrame):
    df.apply(track_prompt_run, axis=1)

def track_prompt_run(prompt_run):
    track.prompt(request_id=str(prompt_run["pl_id"]),
                prompt_name=str(prompt_run["prompt_template_name"]),
                prompt_input_variables={"input": prompt_run["input"]})
    track.score(request_id=str(prompt_run["pl_id"]),
                score=int(match("\d+", prompt_run["score"])[0]))
    