from jinja2 import Template
from agent_prm.agents.agent import Agent
from typing import Callable, List, Dict, Tuple, Any, Optional
from transformers import AutoTokenizer
import requests
from tqdm import tqdm
import json
class SGLangServerAgentWithCritic(Agent):
    """
    An agent that uses sglang and interacts with an sglang server to predict reasons and actions based on task observations and candidate actions. 

    We also use the critic to score the generated reason-action pairs, but the critic does not affect which reason-action pairs are selected. (We always select the reason-action generated with the default temperature)
    """

    def __init__(self, 
                 model_id: str,
                 server_url: str, 
                 prompt_template_file: str, 
                 critic: Any,
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None,
                 max_tokens=256,
                 temperature=0.3,
                 batch_limit=None) -> None:
        """
        Initializes the SGLangServerAgent with the server URL, prompt template, verbosity, and optional debug settings.
        
        Args:
            server_url: The URL of the sglang server (e.g., "http://localhost:30000/").
            prompt_template_file: Path to a Jinja2 template file for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the generated response to extract reason and action.
        """
        self.model_id = model_id
        self.server_url = server_url.rstrip('/') + '/generate'
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_limit = batch_limit

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.critic = critic

    def name(self) -> str:
        return self.model_id

    def predict_reason_action(self, 
                              input_data: Dict) -> Tuple[str, str]:
        """
        Predict reason and action given input_data. 

        Args:
            input_data (Dict): A dictionary containing the necessary data to render the prompt.
        
        Returns:
            A tuple containing the predicted reason (str) and action (str).
            generated_text (str): The generated text from the model.
        """
        raise NotImplementedError("This method is not implemented for SGLangServerAgentWithCritic")
    

    def _request_sglang_server_all_same_temperature(self, input_datas: List[Dict], num_responses: int, temperature: float) -> List[str]:
        """
        Request the sglang server for all the same temperature

        Return: a list of generated texts of len(input_datas) * num_responses
        """
        input_prompts = [self.prompt_template.render(**input_datas[i]) for i in range(len(input_datas)) for _ in range(num_responses)]
        
        conversations = [[{"role": "user", "content": input_prompt}] for input_prompt in input_prompts] # list of lists

        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        generated_texts = []
        
        if self.verbose==1:  
            iterator = tqdm(range(0, len(conversations), batch_limit), desc="Querying sglang agent")
        else:
            iterator = range(0, len(conversations), batch_limit)

        for i in iterator:
            conversations_batch = conversations[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(conversations_batch, tokenize=False, add_generation_prompt=True)
            data_batch = {"model": self.model_id, 
                          "text": prompts_batch,
                          "sampling_params": {
                              "temperature": temperature,
                              "max_new_tokens": self.max_tokens,
                              },
                          }
            responses_batch = requests.post(self.server_url, 
                                            json=data_batch).json()
            generated_texts_batch = [x["text"] for x in responses_batch]
            generated_texts = generated_texts + generated_texts_batch

        return generated_texts
    

    def _request_sglang_server_diff_temperature(self, input_datas: List[Dict], num_responses: int, alt_temperature_for_extra_responses: float) -> List[str]:
        """
        For assume that the first response is the main response (generated using self.temperature) and the rest are alternative responses (so we use alt_temperature_for_extra_responses, e.g., to encourage more diverse responses)

        Parameters:
            input_datas: a list of dictionaries, each containing the necessary data to render the prompt
                size: batch_size
            num_responses: the number of responses to generate
            alt_temperature_for_extra_responses: the temperature to use for the alternative responses

        Returns:
            a list of generated texts of len(input_datas) * num_responses
        """
        main_response_generated_texts = self._request_sglang_server_all_same_temperature(input_datas, num_responses=1, temperature=self.temperature)  # len: batch_size

        # Now, we generate the alternative responses
        alt_response_generated_texts = self._request_sglang_server_all_same_temperature(input_datas, num_responses=num_responses-1, temperature=alt_temperature_for_extra_responses)  # len: batch_size * (num_responses-1)

        # Combine the main and alternative responses (for each query, we want the main response first, then the alternative responses)
        generated_texts = []
        for i in range(len(main_response_generated_texts)):
            generated_texts.append(main_response_generated_texts[i])
            for j in range(num_responses-1):
                generated_texts.append(alt_response_generated_texts[i*(num_responses-1) + j])

        return generated_texts


    def predict_reason_action_batch(self, input_datas: List[Dict], num_responses: int, alt_temperature_for_extra_responses: float = None) -> List[Tuple[str, str]]:
        """
        Return 
            - a list of reason_actions of len(queries), each being len(num_responses)
            - a list of generated texts of len(input_datas) * num_responses
        """
        if alt_temperature_for_extra_responses is None:
            generated_texts = self._request_sglang_server_all_same_temperature(input_datas, num_responses, self.temperature)
        else:
            generated_texts = self._request_sglang_server_diff_temperature(input_datas, num_responses, alt_temperature_for_extra_responses)

        reason_actions_all_queries = []
        counter = 0
        for _ in range(len(input_datas)):
            reason_actions_per_query = []
            for _ in range(num_responses):
                generated_text = generated_texts[counter]
                counter += 1
                # print(f"\n RESPONSE:\n{generated_text}")
                reason, action = self.parse_reason_action_fn(generated_text)
                # print(f"\n REASON:\n{reason}")
                # print(f"\n ACTION:\n{action}")   
                # input("===============")
                reason_actions_per_query.append({'reason': reason, 'action': action})

                if self.verbose >= 2:
                    if self.verbose >= 3: 
                        print(f"\n RESPONSE: {generated_text}")
                    print(f"\n REASON: {reason}")
                    print(f"\n ACTION: {action}")
            reason_actions_all_queries.append(reason_actions_per_query)
        
        # Flatten the queries and their corresponding reason-actions
        flattened_queries_with_reason_action = []
        for query, reason_actions_per_query in zip(input_datas, reason_actions_all_queries):
            for reason_action in reason_actions_per_query:
                flattened_queries_with_reason_action.append({**query, **reason_action})

        # Compute scores for each (query, reason-action) pair
        scores = self.critic.score_reason_action_batch(queries=flattened_queries_with_reason_action)

        # Attach scores to each reason-action
        counter = 0
        for reason_actions_per_query in reason_actions_all_queries:
            for reason_action in reason_actions_per_query:
                reason_action['score'] = scores[counter]
                counter += 1
        
        return reason_actions_all_queries, generated_texts