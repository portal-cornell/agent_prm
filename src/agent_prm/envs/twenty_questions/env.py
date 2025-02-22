"""
Adapted from https://github.com/abdulhaim/LMRL-Gym
"""
from typing import Dict, List, Optional, Tuple
import random
import time
from agent_prm.envs.twenty_questions.data import WordVariants, get_default_word_list
from agent_prm.envs.twenty_questions.simulator import TwentyQuestionsSimulator
from agent_prm.envs.twenty_questions.data import is_done

class TwentyQuestionsEnvironment():
    """
    A state less environment for the 20 questions game.
        (The environment will not track the conversation so far)
    """
    def __init__(
        self, 
        answerer: TwentyQuestionsSimulator,
        word_list: List[WordVariants],  
        max_conversation_length: int=20,
    ):
        self.answerer = answerer
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

        self.random = random.Random(None)
        self.curr_word: Optional[WordVariants] = None

    def step(self, history, action):
        """
        Parameters:
            history (List[Dict]): The history of the conversation so far. A list of dictionaries, of the form:
            {
                "question": str,
                "answer": str,
            }
            action (str): The question to ask the oracle.

        Returns:
            (history, oracle_reason, oracle_answer) (Tuple[List[Dict]], str, str): The updated history of the conversation so far (with the new question and answer)
            reward (float): The reward for the action.
            done (bool): Whether the conversation is done.
        """
        assert self.curr_word is not None, "call env.reset() first."
        
        start_time = time.time()
        answerer_reason, answer = self.answerer.generate_answer(self.curr_word, action)
        end_time = time.time()
        print(f"Time taken to generate answer: {end_time - start_time} seconds")

        # Add to history
        history.append({
            "question": action,
            "answer": answer,
        })

        # Compute the reward for the history
        # Assume that if it's guessing the specific object, it's in the format: "Is it <object>?"
        if "yes" in answer and "no" not in answer and is_done(self.curr_word, action):
            reward = 0.0
            done = True
        else:
            reward = -1.0
            done = False

        if len(history) == self.max_conversation_length:
            print("The word was", self.curr_word[0])
            done = True
        return (history, answerer_reason, answer), reward, done
    
    def reset(self, seed: Optional[int] = None, word = None):
        """
        2 Options to reset the environment:
            1. Reset with a specific word
            2. Reset with a random word (based on the seeds)

        Parameters:
            seed (Optional[int]): The seed to use for the environment.
            options (Optional[Dict]): The options to use for the environment.
            word (Optional[WordVariants]): The word to use for the environment.

        Returns:
            history (List[Dict]): The history of the conversation so far (in the beginning, it's empty). A list of dictionaries, of the form:
            {
                "question": str,
                "answer": str,
            }
        """
        if self.curr_word is not None: 
            print("The word was ", self.curr_word)
            print("Next word...")

        if word is not None:
            assert word in self.word_list, f"Word {word} not in word list."
            self.curr_word = word
        else:
            if seed is not None:
                self.random = random.Random(seed)

            self.curr_word = self.random.choice(self.word_list)

        return []
    

    def copy(self):
        return TwentyQuestionsEnvironment(
            answerer=self.answerer,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )


# TODO: get Batched Environment to work

def setup_twenty_questions_env(data_split: str='all') -> TwentyQuestionsEnvironment:
    env = TwentyQuestionsEnvironment(
        answerer=TwentyQuestionsSimulator(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            prompt_template_file="prompts/20questions/20questions_simulator_template_with-reasoning.j2",
            verbose=1
        ),
        word_list=get_default_word_list(data_split),
        max_conversation_length=20,
    )
    return env