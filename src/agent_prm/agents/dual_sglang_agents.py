from agent_prm.agents.agent import Agent
from typing import List, Dict

class DualSGLangServerAgents(Agent):
    def __init__(self, api_caller: Agent, response_generator: Agent, verbose: int = 0, debug: bool = False):
        self.api_caller = api_caller
        self.response_generator = response_generator
        self.verbose = verbose
        self.debug = debug
        
    def name(self) -> str:
        return f"dual_agents-{self.api_caller.name()}-{self.response_generator.name()}"
    
    def predict_reason_action_batch(self, mode:str, input_datas: List[Dict], num_responses: int, alt_temperature_for_extra_responses: float = None) -> List[List[Dict]]:
        """
        Parameters:
            mode: str, either "api_call" or "generate_response"
            queries: a list of dictionaries, each containing the necessary data to render the prompt
                size: batch_size
            num_responses: the number of responses to generate
            alt_temperature_for_extra_responses: the temperature to use for the alternative responses
                Not implemented. We instead return the top num_responses reason-actions for each query
        """
        if mode == "api_call":
            return self.api_caller.predict_reason_action_batch(input_datas, num_responses, alt_temperature_for_extra_responses)
        elif mode == "generate_response":
            return self.response_generator.predict_reason_action_batch(input_datas, num_responses, alt_temperature_for_extra_responses)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    