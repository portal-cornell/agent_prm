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
    

class BatchedTwentyQuestionsEnvironment(object):
    """
    A more stateless version of the 20 questions environment where we don't keep track of curr_world.

    We will use the trick that our simulator is essentially a simulator, which can accept batched inputs.
    """
    def __init__(self, answerer: TwentyQuestionsSimulator,  word_list: List[WordVariants],  max_conversation_length: int=20):
        self.answerer = answerer
        self.word_list = word_list
        self.max_conversation_length = max_conversation_length

    
    def step(self, words_to_guess: List[WordVariants], histories: List[Dict], actions: List[str], prev_dones: List[bool]):
        """
        Parameters:
            words_to_guess (List[WordVariants]): The secrete words that the agent is trying to guess.
            histories (List[List[Dict]]): The history of the conversation so far (in the beginning, it's empty). A list of lists of dictionaries, of the form:
            [
                [
                    {
                        "question": str,
                        "answer": str,
                    }
                ]
            ]
            actions (List[str]): The actions to take in the environment.
                We assume that even if the conversation is done, there is a placeholder action '' (to make batching easier)
            prev_dones (List[bool]): Whether the conversation is done in the previous step.

        Returns:
            histories (List[List[Dict]]): Updated histories.
            answer_reasons (List[List[str]]): The reasons for the answers.
            answers (List[List[str]]): The answers.
            rewards (List[float]): The rewards for the actions.
            dones (List[bool]): Whether the conversation is done.
        """
        # Get batched answers
        start_time = time.time()
        answer_reasons, answers = self.answerer.generate_answer_batch(words_to_guess, actions)
        end_time = time.time()
        print(f"[ENV] time taken to generate answers: {end_time - start_time} seconds")

        # Update histories
        for i in range(len(histories)):
            if not prev_dones[i]:
                histories[i].append({
                    "question": actions[i],
                    "answer": answers[i],
                })

        rewards = []
        dones = []
        # Compute rewards
        for i in range(len(histories)):
            if prev_dones[i]:
                rewards.append(0.0)
                dones.append(True)
            else:
                if "yes" in answers[i] and "no" not in answers[i] and is_done(words_to_guess[i], actions[i]):
                    reward = 0.0
                    done = True
                else:
                    reward = -1.0
                    done = False
                rewards.append(reward)
                dones.append(done)

            # Check if the agents have exhausted all their guesses
            if len(histories[i]) == self.max_conversation_length:
                dones[i] = True

        return histories, answer_reasons, answers, rewards, dones

    def reset(self, seed: Optional[int] = None, num_envs: int = 1, words_to_guess: Optional[List[WordVariants]] = None):
        """
        Parameters:
            seed (Optional[int]): The seed to use for the environment.
            num_envs (int): The number of environments to create.
            words_to_guess (Optional[List[WordVariants]]): The secrete words that the agent is trying to guess.

        Returns:
            histories (List[List[Dict]]): A list of lists. Each list reprsents the history of the conversation so far (in the beginning, it's empty)
            words_to_guess (List[WordVariants]): The secrete words that the agents are trying to guess.
        """
        if words_to_guess is None:
            words_to_guess = self.random.sample(self.word_list, num_envs)
        else:
            assert len(words_to_guess) == num_envs, "The number of words to guess must be equal to the number of environments."

        histories = [[] for _ in range(num_envs)]

        return histories, words_to_guess

        
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


def setup_batched_twenty_questions_env(data_split: str='all') -> BatchedTwentyQuestionsEnvironment:
    env = BatchedTwentyQuestionsEnvironment(
        answerer=TwentyQuestionsSimulator(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            prompt_template_file="prompts/20questions/20questions_simulator_template_with-reasoning.j2",
            verbose=1
        ),
        word_list=get_default_word_list(data_split),
        max_conversation_length=20,
    )
    return env