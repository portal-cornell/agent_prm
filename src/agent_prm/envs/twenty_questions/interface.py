from typing import List, Dict
import time

from agent_prm.agents.agent import Agent
from agent_prm.envs.twenty_questions.env import BatchedTwentyQuestionsEnvironment
from agent_prm.envs.twenty_questions.data import WordVariants

def query_agent(agent: Agent, history: List[Dict[str, str]], all_obj_list: List[WordVariants], last_question: bool = False):
    """
    Query the agent for a reason and action
    """
    input_data = {
        'mode': 'input' if not last_question else 'input_final',
        'all_obj_list': all_obj_list,
        'observation_action_history': history
    }

    reason, action = agent.predict_reason_action(input_data)

    return reason, action

def query_agent_batch(agent: Agent, histories: List[List[Dict[str, str]]], all_obj_list: List[WordVariants], last_question: bool = List[bool], num_alt_responses: int = 0):
    """
    Query the agent in batch
    """
    input_datas = [
        {
            'mode': 'input' if not last_question[i] else 'input_final',
            'all_obj_list': all_obj_list,
            'observation_action_history': histories[i]
        }
        for i in range(len(histories))
    ]

    reason_actions, generated_texts = agent.predict_reason_action_batch(input_datas, num_responses=1 + num_alt_responses, alt_temperature_for_extra_responses=1.0 if num_alt_responses > 0 else None)

    return reason_actions, generated_texts

def rollout_batch(agent: Agent, batched_env: BatchedTwentyQuestionsEnvironment, all_obj_list: List[WordVariants], words_to_guess: List[WordVariants], histories: List[List[Dict]], traj_list: List[List[Dict]], prev_dones: List[bool], num_alt_responses: int):
    while not all(prev_dones):
        # Batched way
        start_time = time.time()
        reasons, actions, raw_texts, scores = [], [], [], []
        if num_alt_responses > 0:
            alt_reasons, alt_actions, alt_raw_texts, alt_scores = [[] for _ in range(len(histories))], [[] for _ in range(len(histories))], [[] for _ in range(len(histories))], [[] for _ in range(len(histories))]  # For each object, we have a list of alt_reasons and alt_actions and alt_scores
        last_questions = [len(histories[i]) == batched_env.max_conversation_length - 1 for i in range(len(histories))]
        reasons_actions_dict, generated_raw_texts = query_agent_batch(agent, histories, all_obj_list, last_questions, num_alt_responses)
        
        for i in range(len(histories)):
            if prev_dones[i]:
                reasons.append("")
                actions.append("")
                scores.append(None)
                raw_texts.append("")
            else:
                reasons.append(reasons_actions_dict[i][0]["reason"])
                actions.append(reasons_actions_dict[i][0]["action"])
                raw_texts.append(generated_raw_texts[i*(1+num_alt_responses)]) 
                has_critic_score = "score" in reasons_actions_dict[i][0]

                if has_critic_score:
                    scores.append(reasons_actions_dict[i][0]["score"])
                else:
                    scores.append(None)
                
                if num_alt_responses > 0:
                    for j in range(num_alt_responses):
                        alt_reasons[i].append(reasons_actions_dict[i][j+1]["reason"])
                        alt_actions[i].append(reasons_actions_dict[i][j+1]["action"])
                        alt_raw_texts[i].append(generated_raw_texts[i*(1+num_alt_responses) + j + 1])
                        
                        if has_critic_score:
                            alt_scores[i].append(reasons_actions_dict[i][j+1]["score"])
                        else:
                            alt_scores[i].append(None)

        print(f"[AGENT] time taken for batch_size={len(histories)}: {time.time() - start_time}")

        # Step the environment
        histories, answer_reasons, answers, rewards, dones = batched_env.step(words_to_guess, histories, actions, prev_dones)

        # Log the trajectories
        for i in range(len(histories)):
            if not prev_dones[i]:
                traj_list[i].append({
                    "step": len(traj_list[i]),
                    "reason": reasons[i],
                    "action": actions[i],
                    "raw_text": raw_texts[i],
                    "answerer_reason": answer_reasons[i],
                    "answer": answers[i],
                    "reward": rewards[i],
                    "score": scores[i],
                    'alternatives': [
                        {
                            "reason": alt_reasons[i][j],
                            "action": alt_actions[i][j],
                            "raw_text": alt_raw_texts[i][j],
                            "score": alt_scores[i][j]
                        }
                        for j in range(num_alt_responses)
                    ] if num_alt_responses > 0 else None
                })

        prev_dones = dones

    return traj_list