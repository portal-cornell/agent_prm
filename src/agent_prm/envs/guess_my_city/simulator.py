import re
import json
import torch
from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Callable, List, Tuple, Any, Optional, Dict, Union
import requests
from tqdm import tqdm
from typing import Tuple, List
from agent_prm.envs.guess_my_city.data import WordVariants

def parse_reason_and_action_guess_my_city_oracle(text: str) -> Tuple[str, str]:
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
    else:
        reason = "None"
        answer = "None"

    return reason, answer

class GuessMyCitySimulator(object):
    """Initialize the GuessMyCityOracle agent.
    
    - Initialization doesn't change
    - Modify the predict reason action to use the guess my city simulator template
    """
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = parse_reason_and_action_guess_my_city_oracle, 
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
    
    def redact_city_country(self, text: str, city: WordVariants) -> str:
        city_variants = [w.strip().lower() for w in city.words]

        parts = city.words[0].split(",")
        city_name_only = parts[0].strip().lower()
        country_name = parts[1].strip().lower() if len(parts) > 1 else ""

        city_variants.append(city_name_only)
        city_variants = list(set(city_variants)) 

        adjective_map = {
            "south korea": "korean",
            "india": "indian",
            "indonesia": "indonesian",
            "pakistan": "pakistani",
            "turkey": "turkish",
            "china": "chinese",
            "japan": "japanese",
            "thailand": "thai",
            "iran": "iranian",
            "singapore": "singaporean",
            "iraq": "iraqi",
            "bangladesh": "bangladeshi",
            "vietnam": "vietnamese",
            "north korea": "korean",
            "myanmar": "burmese",
            "uzbekistan": "uzbek",
            "philippines": "filipino",
            "brazil": "brazilian",
            "colombia": "colombian",
            "peru": "peruvian",
            "chile": "chilean",
            "argentina": "argentinian",
            "ecuador": "ecuadorian",
            "venezuela": "venezuelan",
            "russia": "russian",
            "uk": "british",
            "united kingdom": "british",
            "germany": "german",
            "spain": "spanish",
            "italy": "italian",
            "ukraine": "ukrainian",
            "france": "french",
            "romania": "romanian",
            "hungary": "hungarian",
            "mexico": "mexican",
            "usa": "american",
            "united states": "american",
            "canada": "canadian",
            "cuba": "cuban",
            "egypt": "egyptian",
            "morocco": "moroccan",
            "congo": "congolese",
            "ethiopia": "ethiopian",
            "cote d'ivorie": "ivoirian",
            "côte d'ivoire": "ivoirian",
            "australia": "australian",
        }

        # First redact all full city names
        for variant in city_variants:
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            text = pattern.sub("[city name redacted]", text)

        if country_name:
            # Redact the full country name
            pattern_country = re.compile(re.escape(country_name), re.IGNORECASE)
            text = pattern_country.sub("[country name redacted]", text)

            # Redact important subwords of country name (e.g., "Korea" from "South Korea")
            country_subwords = re.split(r"[ \-']", country_name)  # Split on space, hyphen, apostrophe
            for subword in country_subwords:
                if len(subword) > 2:  # only redact meaningful words (e.g., ignore 'd' in "Cote d'Ivoire")
                    pattern_subword = re.compile(re.escape(subword), re.IGNORECASE)
                    text = pattern_subword.sub("[country name redacted]", text)

            # Redact adjective form
            adj = adjective_map.get(country_name.lower())
            if adj:
                pattern_adj = re.compile(re.escape(adj), re.IGNORECASE)
                text = pattern_adj.sub("[country name redacted]", text)

                pattern_adj_plural = re.compile(re.escape(adj + "s"), re.IGNORECASE)
                text = pattern_adj_plural.sub("[country name redacted]", text)

        return text


    
    def generate_answer(self, 
                        city: WordVariants, question: str) -> Tuple[str, str]:
        """
        Predicts a reason and an answer given the current city and question

        Args:
            city: The city to generate an answer for
            question: The question to generate an answer for

        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """ 
        input_data = {
            'mode': 'input',
            'city': city[0].lower(),
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
        action = self.redact_city_country(action, city)
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
                              cities: List[WordVariants], 
                              questions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Predicts a reason and an answer given the current city and question
        """
        input_datas = [
            {
                'mode': 'input',
                'city': city[0].lower(),
                'question': question
            }
            for city, question in zip(cities, questions)
        ]

        messages = [
            [
                {"role": "user", "content": self.prompt_template.render(**input_data)}
            ]
            for input_data in input_datas
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


class SGLangServerGuessMyCitySimulator(object):
    """
    Initialize the GuessMyCityOracle agent.
    """
    def __init__(self, 
                 model_id: str, 
                 server_url: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = parse_reason_and_action_guess_my_city_oracle, 
                 max_tokens: int = 256,
                 batch_limit: Optional[int] = None) -> None:
        self.model_id = model_id
        self.server_url = server_url.rstrip('/') + '/generate'
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

        self.max_tokens = max_tokens
        self.batch_limit = batch_limit
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    

    def generate_answer_batch(self, 
                              cities: List[WordVariants], 
                              questions: List[str]) -> Tuple[List[str], List[str]]:
        """
        Predicts a reason and an answer given the current city and question
        """
        input_datas = [
            {
                'mode': 'input',
                'city': city[0].lower(),
                'question': question
            }
            for city, question in zip(cities, questions)
        ]

        messages = [
            [
                {"role": "user", "content": self.prompt_template.render(**input_data)}
            ]
            for input_data in input_datas
        ]

        batch_limit = self.batch_limit if self.batch_limit is not None else len(messages)
        generated_texts = []

        if self.verbose==1:  
            iterator = tqdm(range(0, len(messages), batch_limit), desc="Querying sglang agent")
        else:
            iterator = range(0, len(messages), batch_limit)

        for i in iterator:
            messages_batch = messages[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
            data_batch = {"model": self.model_id, 
                          "text": prompts_batch,
                          "sampling_params": {
                              "temperature": 0.0,
                              "max_new_tokens": self.max_tokens,
                              },
                          }
            
            responses_batch = requests.post(self.server_url, 
                                            json=data_batch).json()
            generated_texts_batch = [x["text"] for x in responses_batch]
            generated_texts.extend(generated_texts_batch)

        answer_reasons, answers = [], []
        for response in generated_texts:
            reason, answer = self.parse_reason_action_fn(response)
            answer_reasons.append(reason)
            answers.append(answer)

        return answer_reasons, answers