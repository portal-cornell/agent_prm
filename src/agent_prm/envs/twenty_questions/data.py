"""
Adapted from https://github.com/abdulhaim/LMRL-Gym
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import nltk

@dataclass
class WordVariants:
    words: List[str]
    pos_tags: List[List[Tuple[str, str]]]

    @classmethod
    def from_list(cls, words_list: List[str]):
        pos_tags = [nltk.pos_tag(nltk.word_tokenize(word.lower())) for word in words_list]
        return cls(words=words_list, pos_tags=pos_tags)

    @classmethod
    def from_str(cls, words_str: str):
        words_list = words_str.split(";")
        return cls.from_list(words_list)

    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        assert 0 <= idx < len(self.words), f"Index {idx} out of range"
        return self.words[idx]

    def json(self):
        return self.words.copy()

    def __str__(self):
        return f"({', '.join(self.words)})"
    
    def __repr__(self) -> str:
        return f"WordVariants([{', '.join(self.words)}])"


DEFAULT_OBJECT_DICT = {
    "Sports": ["Basketball", "Football", "Baseball", "Soccer ball", "Golf ball", "Tennis ball", "Volleyball", "Tennis racket", "Baseball bat", "Helmet"],
    "Animals": ["Cat", "Dog", "Horse", "Cow", "Sheep", "Rabbit", "Lion", "Tiger", "Bear", "Elephant"],
    "Fruits": ["Apple", "Banana", "Orange", "Strawberry", "Grape", "Watermelon", "Pineapple", "Mango", "Cantaloupe", "Peach"],
    "Vehicles": ["Car", "Truck", "Motorcycle", "Boat", "Airplane;Plane", "Train", "Bus", "Helicopter", "Scooter", "Ship"],
    "Clothes": ["Shirt", "Pants;Pant;Pair of pants", "Jacket", "Dress", "Skirt", "Belt", "Shoes;Shoe;Pair of shoes", "Boots;Boot;Pair of boots", "Socks;Sock;Pair of socks", "Hat", "Scarf"],
    "Electronics": ["Computer", "Smartphone", "Television;TV", "Headphone;Headphones;Pair of headphones", "Monitor;Computer monitor", "Camera", "Microwave;Microwave oven", "Refrigerator", "Blender", "Computer keyboard;Keyboard"],
    "Musical Instruments": ["Piano", "Guitar", "Drum;Drums", "Violin", "Saxophone", "Flute", "Trumpet", "Clarinet", "Harp", "Trombone"],
    "Furniture": ["Chair", "Table", "Bed", "Desk", "Couch", "Dresser", "Bookcase", "Nightstand", "Mattress", "Pillow"],
    "Office Supplies": ["Pen", "Paper;Piece of paper", "Stapler", "Printer", "Calculator", "Battery;Battery pack;Pack of batteries", "Toothbrush", "Toothpaste", "Pencil", "Sharpie", "Scissors;Pair of scissors", "Key", "Diary", "Calendar"],
    "Vegetables": ["Carrot", "Potato", "Broccoli", "Tomato", "Onion", "Spinach", "Corn", "Peas;Pea", "Celery", "Cucumber"],
    "Art": ["Painting;Canvas painting;Oil painting;Watercolor painting", "Paintbrush", "Canvas;Painting canvas", "Eraser;Pencil eraser", "Marker", "Glue;Glue stick;Bottle of glue", "Sculpture"],
    "Kitchen Tools": ["Knife", "Spoon", "Fork", "Plate", "Bowl", "Cooking pot;Pot", "Pan;Saucepan;Frying pan", "Cup", "Chopstick;Chopsticks;Pair of chopsticks", "Whisk"],
    "Nature": ["Rock", "Tree", "Bush", "Mountain", "Forest", "Ocean", "Sea", "Lake", "River", "Meteorite", "Cactus"],
    "Toys": ["Lego;Lego set", "Doll;Toy doll;Plush doll", "Kite", "Puzzle;Jigsaw puzzle", "Stuffed animal"],
    "Jewelry": ["Earring;Earrings;Pair of earrings", "Necklace", "Bracelet", "Ring", "Brooch", "Hairclip", "Pendant", "Watch", "Locket"],
    "Garden Supplies": ["Gloves;Glove;Pair of gloves", "Shovel", "Rake", "Watering can", "Lawn mower"],
    "Tools": ["Hammer", "Screwdriver", "Wrench", "Saw", "Pliers;plier;Pair of pliers", "Drill"]
}

TRAIN_OBJECT_DICT = {
    'Sports': [
        'Basketball', 'Football', 'Baseball', 'Golf ball', 'Tennis ball',
        'Volleyball', 'Tennis racket', 'Helmet'
    ],
    'Animals': [
        'Cat', 'Dog', 'Horse', 'Cow', 'Rabbit', 'Lion', 'Tiger', 'Bear'
    ],
    'Fruits': [
        'Apple', 'Banana', 'Orange', 'Strawberry', 'Grape', 'Mango',
        'Cantaloupe', 'Peach'
    ],
    'Vehicles': [
        'Car', 'Truck', 'Airplane;Plane', 'Train', 'Bus', 'Helicopter',
        'Scooter', 'Ship'
    ],
    'Clothes': [
        'Shirt', 'Pants;Pant;Pair of pants', 'Jacket', 'Dress', 'Skirt',
        'Belt', 'Shoes;Shoe;Pair of shoes', 'Socks;Sock;Pair of socks', 'Hat'
    ],
    'Furniture': [
        'Chair', 'Bed', 'Desk', 'Couch', 'Dresser', 'Bookcase',
        'Nightstand', 'Pillow'
    ],
    'Office Supplies': [
        'Pen', 'Paper;Piece of paper', 'Stapler', 'Printer', 'Calculator',
        'Toothbrush', 'Pencil', 'Scissors;Pair of scissors', 'Key',
        'Diary', 'Calendar'
    ],
    'Vegetables': [
        'Potato', 'Broccoli', 'Tomato', 'Onion', 'Corn', 'Peas;Pea',
        'Celery', 'Cucumber'
    ],
    'Art': [
        'Painting;Canvas painting;Oil painting;Watercolor painting',
        'Paintbrush', 'Marker', 'Glue;Glue stick;Bottle of glue', 'Sculpture'
    ],
    'Kitchen Tools': [
        'Knife', 'Fork', 'Plate', 'Cooking pot;Pot',
        'Pan;Saucepan;Frying pan', 'Cup', 'Chopstick;Chopsticks;Pair of chopsticks',
        'Whisk'
    ],
    'Nature': [
        'Tree', 'Bush', 'Forest', 'Ocean', 'Sea', 'Lake', 'River',
        'Meteorite', 'Cactus'
    ],
    'Toys': [
        'Lego;Lego set', 'Kite', 'Puzzle;Jigsaw puzzle', 'Stuffed animal'
    ],
    'Jewelry': [
        'Earring;Earrings;Pair of earrings', 'Necklace', 'Ring',
        'Brooch', 'Pendant', 'Watch', 'Locket'
    ],
    'Garden Supplies': [
        'Gloves;Glove;Pair of gloves', 'Shovel', 'Watering can',
        'Lawn mower'
    ],
    'Tools': [
        'Screwdriver', 'Wrench', 'Saw', 'Pliers;plier;Pair of pliers', 'Drill'
    ]
}

# Under the same category as the training objects, but not in the training objects
VALIDATION_OBJECT_DICT = data = {
    'Sports': [
        'Baseball bat', 'Soccer ball'
    ],
    'Animals': [
        'Elephant', 'Sheep'
    ],
    'Fruits': [
        'Watermelon', 'Pineapple'
    ],
    'Vehicles': [
        'Motorcycle', 'Boat'
    ],
    'Clothes': [
        'Scarf', 'Boots;Boot;Pair of boots'
    ],
    'Furniture': [
        'Mattress', 'Table'
    ],
    'Office Supplies': [
        'Battery;Battery pack;Pack of batteries', 'Sharpie', 'Toothpaste'
    ],
    'Vegetables': [
        'Carrot', 'Spinach'
    ],
    'Art': [
        'Eraser;Pencil eraser', 'Canvas;Painting canvas'
    ],
    'Kitchen Tools': [
        'Spoon', 'Bowl'
    ],
    'Nature': [
        'Mountain', 'Rock'
    ],
    'Toys': [
        'Doll;Toy doll;Plush doll'
    ],
    'Jewelry': [
        'Bracelet', 'Hairclip'
    ],
    'Garden Supplies': [
        'Rake'
    ],
    'Tools': [
        'Hammer'
    ]
}

TEST_OBJECT_DICT = {
    'Musical Instruments': [
        'Piano', 'Guitar', 'Drum;Drums', 'Violin', 'Saxophone', 
        'Flute', 'Trumpet', 'Clarinet', 'Harp', 'Trombone'
    ],
    'Electronics': [
        'Computer', 'Smartphone', 'Television;TV', 
        'Headphone;Headphones;Pair of headphones', 
        'Monitor;Computer monitor', 'Camera', 
        'Microwave;Microwave oven', 'Refrigerator', 
        'Blender', 'Computer keyboard;Keyboard'
    ]
}


INVALID_QUESTION = "Is this a valid question?\n"
INITIAL_STR = "Questions:\n"


def get_default_word_list(data_split: str = "all") -> List[WordVariants]:
    if data_split == "all":
        dict_to_use = DEFAULT_OBJECT_DICT
    elif data_split == "train":
        dict_to_use = TRAIN_OBJECT_DICT
    elif data_split == "val":
        dict_to_use = VALIDATION_OBJECT_DICT
    elif data_split == "test":
        dict_to_use = TEST_OBJECT_DICT
    else:
        raise ValueError(f"Invalid data split: {data_split}")
    
    word_list = []
    for _, words in dict_to_use.items():
        word_list.extend(map(lambda x: WordVariants.from_str(x), words))
    return word_list


def is_done(word_var: WordVariants, question: str):
    # cut out punctuations at the end
    while len(question) > 0 and not question[-1].isalpha():
        question = question[:-1]

    if len(question) == 0:
        return False

    question_pos = nltk.pos_tag(nltk.word_tokenize(question.lower()))

    # ignore these nouns when checking for extra words
    ignores = {"object", "something", "type", "kind", "entity", "secret", "object/entity"}
    for pos_list in word_var.pos_tags:
        for w, _ in pos_list:
            ignores.add(w)

    # check for extra words
    for q_i in range(len(question_pos)):
        q_i_word, q_i_pos = question_pos[q_i]

        # check if the current word is a noun that shouldn't be ignored
        if q_i_pos[:2] == "NN" and q_i_word not in ignores:
            # if it's a counter word that comes before "of", also ignore it
            if q_i + 1 < len(question_pos) and question_pos[q_i + 1][0] == "of":
                continue
            # extra word found
            return False

    # check for the actual word at the end of the question
    for word_pos in word_var.pos_tags:
        if len(word_pos) > len(question_pos):
            continue
        
        all_same = True
        for (var_i_word, _), (q_i_word, _) in zip(word_pos, question_pos[-len(word_pos):]):
            if var_i_word != q_i_word:
                all_same = False
                break
        if all_same:
            return True
    
    return False