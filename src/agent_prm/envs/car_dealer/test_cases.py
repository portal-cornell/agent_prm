"""
python src/agent_prm/envs/car_dealer/test_cases.py
"""

from agent_prm.envs.car_dealer.data import *
from agent_prm.envs.car_dealer.parser import parse_reason_and_action_car_dealer_api_call, parse_reason_and_action_car_dealer
import json

# ### Testing get_price_comparison
# proposed_car = {}
# x0 = "For the exact car you want, I can only give you discount to $60000. It was $70000 before so this is really good for you!"
# print(get_price_comparison(buyer_info={"budget": 50000, "msrp": 70000}, seller_response=x0, seller_proposed_car={}))
# input("x0")
# proposed_car = {}
# x1 = "For the exact car you want, its original price is $70000. I can only give you discount to $60000. This is a really good deal for you!"
# print(get_price_comparison(buyer_info={"budget": 50000, "msrp": 70000}, seller_response=x1, seller_proposed_car={}))
# input("x1 (flip order)")
# # Price is lower than the original price that buyer was interested in. But the seller hasn't offered a discount for the car that is being proposed.
# proposed_car = {"msrp": 60000}
# x2 = "I found another car that is $60000. This is a really good deal for you!"
# print(get_price_comparison(buyer_info={"budget": 50000, "msrp": 70000}, seller_response=x2, seller_proposed_car=proposed_car))
# input("x2")
# # Offered a discount for the car that is being proposed.
# proposed_car = {"msrp": 60000}
# x3 = "I found another car that is $60000, but I can give it to you at $50000 to match your budget. This is a really good deal for you!"
# print(get_price_comparison(buyer_info={"budget": 50000, "msrp": 70000}, seller_response=x3, seller_proposed_car=proposed_car))
# input("x3")


# # ### Testing extract_final_decision_from_buyer_reply
# regular conversation
# x0 = "I'm interested in a car with a navigation system and a backup camera. I'm willing to pay up to $50,000."
# print(extract_final_decision_from_buyer_reply(x0))
# input("x0")
# # Car bought
# x1 = "Decision=Accept Buy_Price=$50000"
# print(extract_final_decision_from_buyer_reply(x1))
# input("x1")
# # Car not bought
# x2 = "Decision=Reject Buy_Price=$0"
# print(extract_final_decision_from_buyer_reply(x2))
# input("x2")
# # White space
# x3 = "Decision =   Accept     Buy_Price  =  $ 50000"
# print(extract_final_decision_from_buyer_reply(x3))
# input("x3")
# x4 = "Decision=Accept, Buy_Price=$54000"
# print(extract_final_decision_from_buyer_reply(x4))
# input("x4")

### Testing compute_reward
# B1 success (Gave a discount that's matching the user's budget)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B1,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": [],
#         "budget": 50000,
#         "msrp": 55000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 50000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": [],
#         "msrp": 55000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B1 success (gave a discount that's matching the user's budget)")
# # B1 success (different car and under budget)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B1,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["backup camera", "navigation system"],
#         "budget": 50000,
#         "msrp": 65000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 50000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "sedan",
#         "features": [],
#         "msrp": 65000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B1 success (different car and under budget)")
# # B1 success (discount but over budget)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B1,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["sunroof", "Apple CarPlay"],
#         "budget": 50000,
#         "msrp": 65000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 55000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": ["sunroof", "Apple CarPlay"],
#         "msrp": 65000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B1 success (discount but over budget)")
# # B1 failed (car is at budget but no discount)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B1,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["sunroof", "Apple CarPlay"],
#         "budget": 65000,
#         "msrp": 65000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 65000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": ["sunroof", "Apple CarPlay"],
#         "msrp": 65000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert not success
# input("B1 failed (car is at budget but no discount)")
# # B2 success (car has all the features, and the budget is exactly the car price)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B2,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["sunroof", "Apple CarPlay"],
#         "budget": 65000,
#         "msrp": 65000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 65000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": ["Apple CarPlay", "sunroof"],
#         "msrp": 65000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B2 success (car has all the features, and the budget is exactly the car price)")
# # B2 success (car has all the features, and the budget is over the car price)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B2,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["sunroof", "Apple CarPlay"],
#         "budget": 65000,
#         "msrp": 67000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 67000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": ["Apple CarPlay", "sunroof"],
#         "msrp": 67000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B2 success (car has all the features, and the budget is over the car price)")
# # B2 failed (car doesn't have all the features, but the budget is under the car price)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B2,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["backup camera", "navigation system"],
#         "budget": 65000,
#         "msrp": 75000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 63000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": ["backup camera"],
#         "msrp": 66000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert not success
# input("B2 failed (car doesn't have all the features, but the budget is under the car price)")
# # B4 success (car is under the budget AND the brand is the preferred brand, even though the car is not the preferred type)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B4,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": ["backup camera", "navigation system"],
#         "budget": 45000,
#         "msrp": 80000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 45000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "sedan",
#         "features": [],
#         "msrp": 45000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B4 success (car is under the budget AND the brand is the preferred brand)")
# # B4 failed (car is over the budget)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B4,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": [],
#         "budget": 45000,
#         "msrp": 80000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 55000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": [],
#         "msrp": 80000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert not success
# input("B4 failed (car is over the budget)")
# # B4 failed (car is not the preferred brand)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B4,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": [],
#         "budget": 45000,
#         "msrp": 80000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 43000
#     },
#     curr_car_of_interest={
#         "brand": "Toyota",
#         "type": "van",
#         "features": [],
#         "msrp": 43000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert not success
# input("B4 failed (car is not the preferred brand)")
# # B5 success (car is under the budget AND the type is the preferred type, even though the car is not the preferred brand)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B5,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": [],
#         "budget": 45000,
#         "msrp": 55000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 43000
#     },
#     curr_car_of_interest={
#         "brand": "Toyota",
#         "type": "van",
#         "features": [],
#         "msrp": 45000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert success
# input("B5 success (car is under the budget AND the type is the preferred type, even though the car is not the preferred brand)")
# # B5 failed (car is over the budget)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B5,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": [],
#         "budget": 45000,
#         "msrp": 55000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 55000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "van",
#         "features": [],
#         "msrp": 55000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert not success
# input("B5 failed (car is over the budget)")
# # B5 failed (car is not the preferred type)
# r, success = compute_reward(
#     buyer_info={
#         "buyer_strategy": B5,
#         "preferred_brand": "Volkswagen",
#         "preferred_type": "van",
#         "preferred_features": [],
#         "budget": 45000,
#         "msrp": 55000
#     },
#     final_decision={
#         "car_bought": True,
#         "buy_price": 44000
#     },
#     curr_car_of_interest={
#         "brand": "Volkswagen",
#         "type": "sedan",
#         "features": [],
#         "msrp": 55000
#     }
# )
# print(f"r: {r}, success: {success}")
# assert not success
# input("B5 failed (car is not the preferred type)")

# from agent_prm.envs.car_dealer.parser import parse_reason_and_action_car_dealer_api_call
# import json
# # x0 correct format
# x0 = """REASON:
# I want to search for a car with a navigation system and a backup camera.
# API NAME:
# search_car_by_brand_type
# API BRAND:
# Volkswagen
# API TYPE:
# van"""
# reason, action = parse_reason_and_action_car_dealer_api_call(x0)
# print(f"reason: {reason}")
# print(f"action:\n{json.dumps(action, indent=4)}")
# input("x0")

# # x1 more spaces 
# x1 = """REASON:
# I want to search for a car with a navigation system and a backup camera.

#     API NAME:
#         search_car_by_brand_type

# API BRAND:
#         Volkswagen
# API TYPE: van"""
# reason, action = parse_reason_and_action_car_dealer_api_call(x1)
# print(f"reason: {reason}")
# print(f"action:\n{json.dumps(action, indent=4)}")
# input("x1")

# # x2 no api name
# x2 = """REASON:
# I want to search for a car with a navigation system and a backup camera.
# """
# reason, action = parse_reason_and_action_car_dealer_api_call(x2)
# print(f"reason: {reason}")
# print(f"action:\n{json.dumps(action, indent=4)}")
# input("x2 (no api name) - Should fail")

# # x3 no new line
# x3 = """REASON: I want to search for a car with a navigation system and a backup camera.
# API NAME: search_car_by_brand_type
# API BRAND: Volkswagen
# API TYPE: van"""
# reason, action = parse_reason_and_action_car_dealer_api_call(x3)
# print(f"reason: {reason}")
# print(f"action:\n{json.dumps(action, indent=4)}")
# input("x3 (no new line)")

# x4 no space
x4 = """REASON:I want to search for a car with a navigation system and a backup camera.
API NAME:search_car_by_brand_type
API BRAND:Volkswagen
API TYPE:van
API FEATURES:[]"""
reason, action = parse_reason_and_action_car_dealer_api_call(x4)
print(f"reason: {reason}")
print(f"action:\n{json.dumps(action, indent=4)}")
print(parse_reason_and_action_car_dealer(x4))
input("x4 (no space)")

# x5 wrong api name
x5 = """REASON: I want to search for a car with a navigation system and a backup camera.
API NAME: search_car_by_brand_type_features
API BRAND: Volkswagen
API TYPE: van
API FEATURES:[]"""
reason, action = parse_reason_and_action_car_dealer_api_call(x5)
print(f"reason: {reason}")
print(f"action:\n{json.dumps(action, indent=4)}")
print(parse_reason_and_action_car_dealer(x5))
input("x5 (wrong api name) - Should fail")