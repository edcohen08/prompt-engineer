from typing import Dict, List, Optional

from langchain.chains import LLMChain, SequentialChain
from langchain.schema import LLMResult


class PromptLayerSequentialChain(SequentialChain):
    return_pl_ids: bool = False
    
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        known_values = inputs.copy()
        return_pl_ids = False
        for i, chain in enumerate(self.chains):
            return_pl_ids = getattr(chain.llm, "return_pl_id", False) or return_pl_ids
            outputs = chain(known_values)
            known_values.update(outputs)
        outputs = {k: known_values[k] for k in self.output_variables}
        if return_pl_ids:
            outputs.update({"pl_ids": known_values["pl_ids"]})

        self.return_pl_ids = return_pl_ids
        return outputs
    
    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        output_keys_set = set(self.output_keys)
        if self.return_pl_ids:
            output_keys_set.add("pl_ids")
        
        if set(outputs) != output_keys_set:
            raise ValueError(
                f"Did not get output keys that were expected. "
                f"Got: {set(outputs)}. Expected: {output_keys_set}.")
    
class PromptLayerLLMChain(LLMChain):
    pl_ids = Optional[List[Dict[str, str]]]

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        outputs = [
            # Get the text of the top generated string.
            {self.output_key: generation[0].text}
            for generation in response.generations
        ]
        if getattr(self.llm, "return_pl_id", False):
            self.pl_ids = [{"pl_id": generation[0].generation_info["pl_request_id"]} for generation in response.generations]
        
        return outputs
    
    def prep_outputs(self, inputs: Dict[str, str], outputs: Dict[str, str], return_only_outputs: bool = False) -> Dict[str, str]:
        outputs = super().prep_outputs(inputs, outputs, return_only_outputs)
        if getattr(self.llm, "return_pl_id", False):
            outputs.update({"pl_ids": self.pl_ids})

        return outputs

        

            
