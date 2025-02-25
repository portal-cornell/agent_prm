import re
import json
import torch
from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Callable, List, Tuple, Any, Optional, Dict, Union

from typing import Tuple, List
from agent_prm.envs.twenty_questions.data import WordVariants

def parse_reason_and_action_20questions_oracle(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for ORCALE environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action. 
    """
    pattern = r"REASON:\s*([\s\S]*?)\s*ANSWER:\s*([\s\S]*?)$"
    match = re.search(pattern, text)

    if match:
        reason = match.group(1).strip()  # Remove extra spaces/newlines
        answer = match.group(2).strip()

        # Clean up action to move to lower case and remove any random characters
        answer = answer.lower()
        answer = re.sub(r'[^a-z0-9 /]', '', answer)
    else:
        reason = "None"
        answer = "None"

    if answer != "yes" and answer != "no":
        answer = "no"

    return reason, answer

class TwentyQuestionsSimulator(object):
    """Initialize the TwentyQuestionsOracle agent.
    
    - Initialization doesn't change
    - Modify the predict reason action to use the twenty questions simulator template
    """
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = parse_reason_and_action_20questions_oracle, 
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
        self.parse_reason_action_fn = parse_reason_action_fn
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
    
    def generate_answer(self, 
                        word: WordVariants, question: str) -> Tuple[str, str]:
        """
        Predicts a reason and an asnwer given the current word and question

        Args:
            words: The word to generate an answer for
            question: The question to generate an answer for

        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """ 
        input_data = {
            'mode': 'input',
            'thing': word[0].lower(),
            'question': question
        }
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
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output = outputs[0]
        
        response = self.tokenizer.decode(output[tokenized_inputs["input_ids"].shape[-1] :],skip_special_tokens=True)
        reason, action = self.parse_reason_action_fn(response)
        if self.verbose > 0:
            print(f"------ env simulator ------")
            print(f" REASON: {reason}")
            print(f" ANSWER: {action}")
            print(f"------ env simulator ------")
        if self.debug:
            human_input = input()
            if human_input != "c":
                action = human_input
                reason = "None"

        return reason, action
    

    def generate_answer_batch(self, 
                              words: List[WordVariants], 
                              questions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Predicts a reason and an asnwer given the current word and question
        """
        input_datas = [
            {
                'mode': 'input',
                'thing': word[0].lower(),
                'question': question
            }
            for word, question in zip(words, questions)
        ]

        messages = [
            self.prompt_template.render(**input_data)
            for input_data in input_datas
        ]

        tokenized_inputs = self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)  # size for "input_ids" is (bs, seq_len)

        outputs = self.model.generate(
            **tokenized_inputs,
            max_new_tokens=256,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )  # size: (bs, seq_len)

        # We want to only decode the last part of the output
        responses = self.tokenizer.batch_decode(outputs[:, tokenized_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        answer_reasons, answers = [], []
        for response in responses:
            reason, answer = self.parse_reason_action_fn(response)
            answer_reasons.append(reason)
            answers.append(answer)

        return answer_reasons, answers
