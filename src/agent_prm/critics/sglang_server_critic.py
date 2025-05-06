from agent_prm.critics.critic import Critic
from jinja2 import Template
from transformers import AutoTokenizer, AutoConfig
from typing import Callable, List, Tuple, Dict, Any, Optional
import requests
from tqdm import tqdm
import json
class SGLangServerCritic(Critic):
    def __init__(self, 
                 model_id: str, 
                 server_url: str, 
                 prompt_template_file: str, 
                 include_reason: bool = True,
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None,
                 format_reason_action_fn: Callable[[str], str] = None,
                 batch_limit = None) -> None:
        self.server_url = server_url.rstrip('/') + '/classify'
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        self.include_reason = include_reason
        self.set_prompt_template(prompt_file_path=prompt_template_file)

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
        self.tokenizer = tokenizer
        self.batch_limit = batch_limit
        self.format_reason_action_fn = format_reason_action_fn
    
    def name(self) -> str:
        return self.model_id

    def set_prompt_template(self, prompt_file_path: str = "", prompt_template: Template = None):
        if prompt_file_path != "" and prompt_template is None:
            with open(prompt_file_path, "r") as file:
                self.prompt_template = Template(file.read())
        elif prompt_file_path != "" and prompt_template is not None:
            raise ValueError("Cannot provide both prompt_file_path and prompt_template")
        else:
            self.prompt_template = prompt_template

    def score_reason_action_batch(self, queries: List[Dict]) -> List[float]:
        conversations = []
        
        i = 0
        for query in queries:
            # We assume that the query already has the mode
            input_prompt = self.prompt_template.render(**query).strip()

            if self.format_reason_action_fn is not None:
                output_data = self.format_reason_action_fn(query)
            else:
                output_data = query
            
            output_data["mode"] = "output" if self.include_reason else "output_no_reason"
            output_prompt = self.prompt_template.render(**output_data).strip()

            conversations.append([{"role": "user", "content": input_prompt}, {"role": "assistant", "content": output_prompt}])      
            i += 1

        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        scores = []
        for i in range(0, len(conversations), batch_limit):
            conversations_batch = conversations[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(conversations_batch, tokenize=False)
            data_batch = {"model": self.model_id, "text": prompts_batch}
            responses_batch = requests.post(self.server_url, json=data_batch).json()
            scores_batch = [x["embedding"][0] for x in responses_batch]
            scores = scores + scores_batch
        
        return scores
    
    def score_state_batch(self, queries: List[Dict]) -> List[float]:
        conversations = []
        for query in queries:
            observation_action_history = [
                {"observation": entry["observation"], "action": entry["action"]}
                for entry in query["observation_action_history"]
            ]
            input_data = {
                "mode": "input",
                "task": query["task"],
                "observation": query["observation"],
                "candidate_actions": query["candidate_actions"],
                "observation_action_history": observation_action_history,      
            }
            input_prompt = self.prompt_template.render(**input_data)

            conversations.append([{"role": "user", "content": input_prompt}])      

        batch_limit = self.batch_limit if self.batch_limit is not None else len(conversations)
        scores = []
        
        if self.verbose==1:  
            iterator = tqdm(range(0, len(conversations), batch_limit), desc="Querying sglang critic")
        else:
            iterator = range(0, len(conversations), batch_limit)

        for i in iterator:
            conversations_batch = conversations[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(conversations_batch, tokenize=False)
            data_batch = {"model": self.model_id, "text": prompts_batch}
            responses_batch = requests.post(self.server_url, json=data_batch).json()
            scores_batch = [x["embedding"][0] for x in responses_batch]
            scores = scores + scores_batch
            
        return scores