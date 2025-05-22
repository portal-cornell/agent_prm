"""
python src/agent_prm/envs/twenty_questions/test_cases.py
"""
from agent_prm.utils.parser import parse_hindsight_reason_and_action_twenty_questions

# x0 proper format
x0 = """
TEACHER_REASON:
I want to search for a car with a navigation system and a backup camera.
QUESTION:
What is the brand of the car?
PLAYER_REASON:
I want to search for a car with a navigation system and a backup camera.
"""
reason, action = parse_hindsight_reason_and_action_twenty_questions(x0)
print(f"reason: {reason}")
print(f"action:\n{action}")
input("x0 (proper format)")

# x1 no punctuation
x1 = """
TEACHER_REASON
I want to search for a car with a navigation system and a backup camera.
QUESTION
What is the brand of the car?
PLAYER_REASON
I want to search for a car with a navigation system and a backup camera.
"""
reason, action = parse_hindsight_reason_and_action_twenty_questions(x1)
print(f"reason: {reason}")
print(f"action:\n{action}")
input("x1 (no punctuation)")

# x2 no space
x2 = """TEACHER_REASON:I want to search for a car with a navigation system and a backup camera.
QUESTION:What is the brand of the car?
PLAYER_REASON:I want to search for a car with a navigation system and a backup camera.
"""
reason, action = parse_hindsight_reason_and_action_twenty_questions(x2)
print(f"reason: {reason}")
print(f"action:\n{action}")
input("x2 (no space)")

# x3 no question
x3 = """
TEACHER_REASON:
I want to search for a car with a navigation system and a backup camera.
PLAYER_REASON:
I want to search for a car with a navigation system and a backup camera.
"""
reason, action = parse_hindsight_reason_and_action_twenty_questions(x3)
print(f"reason: {reason}")
print(f"action:\n{action}")
input("x3 (no question)")

# x4 no teacher reason
x4 = """
QUESTION:
What is the brand of the car?
PLAYER_REASON:
I want to search for a car with a navigation system and a backup camera.
"""
reason, action = parse_hindsight_reason_and_action_twenty_questions(x4)
print(f"reason: {reason}")
print(f"action:\n{action}")
input("x4 (no teacher reason)")

# x5 no player reason
x5 = """
TEACHER_REASON:
I want to search for a car with a navigation system and a backup camera.
QUESTION:
What is the brand of the car?
"""
reason, action = parse_hindsight_reason_and_action_twenty_questions(x5)
print(f"reason: {reason}")
print(f"action:\n{action}")
input("x5 (no player reason)")