import torch
from jinja2 import Template
from agent_prm.agents.agent import Agent
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable, List, Tuple, Any, Optional, Dict

class HFAgent(Agent):
    """
    A Hugging Face-based agent that utilizes a pre-trained language model to predict reasons 
    and actions based on observations and candidate actions.
    """
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None, 
                 max_length: Optional[int] = None) -> None:
        """
        Initializes the HFAgent with a pre-trained language model, tokenizer, and a prompt template.

        Args:
            model_id: The identifier for the Hugging Face model.
            prompt_template_file: Path to a Jinja2 template file used to create input prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the model's response to extract reason and action.
            max_length: An optional maximum length for tokenization (default is None).
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = model
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.set_parse_reason_action_fn(parse_reason_action_fn)
        self.set_prompt_template(prompt_file_path=prompt_template_file)
        self.max_length = max_length  
        tokenizer = AutoTokenizer.from_pretrained(model_id, truncation=True, padding=True)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

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

    def set_parse_reason_action_fn(self, parse_reason_action_fn: Callable[[str], Tuple[str, str]]):
        self.parse_reason_action_fn = parse_reason_action_fn
    
    def predict_reason_action(self, 
                              input_data: Dict) -> Tuple[str, str]:
        """
        Predicts a reason and an action given input_data.

        Args:
            input_data (Dict): A dictionary containing the necessary data to render the prompt.

        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """ 
        input_prompt = self.prompt_template.render(**input_data)
        
        messages = [
            {"role": "user", "content": input_prompt}
        ]
        message = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized_inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)
        
        outputs = self.model.generate(
            tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            max_new_tokens=256,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output = outputs[0]
        
        response = self.tokenizer.decode(output[tokenized_inputs["input_ids"].shape[-1] :],skip_special_tokens=True)

        reason, action = self.parse_reason_action_fn(response)
        if self.verbose > 0:
            print(f"\n REASON: {reason}")
            print(f"\n ACTION: {action}")
        
        if self.debug:
            human_input = input()
            if human_input != "c":
                action = human_input
                reason = "None"

        return reason, action

    # This is not done in an actual batch way (since it's still using a for loop)
    # def predict_reason_action_batch(self, input_datas: List[Dict], num_responses: int) -> List[Tuple[str, str]]:
    #     """
    #     Return a list of reason_actions of len(queries), each being len(num_responses)
    #     """
    #     reason_actions_all_queries = []
    #     for i in range(len(input_datas)):
    #         reason_actions_per_query = []
    #         for _ in range(num_responses):
    #             reason, action = self.predict_reason_action(input_datas[i])
    #             reason_actions_per_query.append({'reason': reason, 'action': action})
    #         reason_actions_all_queries.append(reason_actions_per_query)

    #     return reason_actions_all_queries
    
    def predict_reason_action_batch(self, input_datas: List[Dict], num_responses: int, alt_temperature_for_extra_responses: float = None) -> List[Tuple[str, str]]:
        """
        Return a list of reason_actions of len(queries), each being len(num_responses)

        Currently, not supporting different temperatures for different responses.
        """
        messages = [
            [
                {"role": "user", "content": self.prompt_template.render(**input_data)}
            ]
            for input_data in input_datas for _ in range(num_responses)
        ]

        messages = [
            self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        tokenized_inputs = self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)  # size for "input_ids" is (bs, seq_len)

        outputs = self.model.generate(
            **tokenized_inputs,
            max_new_tokens=256,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id
        )  # size: (bs, seq_len)

        # We want to only decode the last part of the output
        responses = self.tokenizer.batch_decode(outputs[:, tokenized_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        reason_actions_all_queries = []
        counter = 0
        for i in range(len(responses)):
            if counter == 0:
                reason_actions_per_query = []
            
            reason, action = self.parse_reason_action_fn(responses[i])
            reason_actions_per_query.append({'reason': reason, 'action': action})
            counter += 1

            if counter == num_responses:
                reason_actions_all_queries.append(reason_actions_per_query)
                counter = 0

        return reason_actions_all_queries, responses
            