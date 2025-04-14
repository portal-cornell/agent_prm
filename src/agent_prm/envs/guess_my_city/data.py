"""
Adapted from https://github.com/abdulhaim/LMRL-Gym
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import nltk
import re

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


INVALID_QUESTION = "Is this a valid question?\n"
INITIAL_STR = "Questions:\n"

DEFAULT_CITY_DICT = {
    "South Korea": ["Seoul, South Korea", "Pusan, South Korea", "Taega, South Korea", "Inchon, South Korea"],
    "Brazil": ["Sao Paulo, Brazil", "Rio de Janeiro, Brazil", "Salvador, Brazil", "Belo Horizonte, Brazil", "Fortaleza, Brazil", "Brasilia, Brazil"],
    "India": ["Bombay, India", "Delhi, India", "Calcutta, India", "Madras, India", "Bangalore, India", "Hyderabad, India", "Ahmedabad, India", "Kanpur, India"],
    "Indonesia": ["Jakarta, Indonesia", "Bandung, Indonesia", "Bagor, Indonesia", "Malang, Indonesia", "Surabaya, Indonesia", "Semarang, Indonesia", "Sukabumi, Indonesia", "Cirebon, Indonesia", "Medan, Indonesia"],
    "Pakistan": ["Karachi, Pakistan", "Lahore, Pakistan", "Faisalabad, Pakistan"],
    "Russia": ["Moscow, Russia", "St Petersburg, Russia"],
    "Turkey": ["Istanbul, Turkey", "Ankara, Turkey", "Izmir, Turkey"],
    "Mexico": ["Mexico City, Mexico"],
    "China": ["Shanghai, China", "Beijing, China", "Hong Kong, China", "Tianjin, China", "Shenyang, China", "Wuhan, China", "Guangzhou, China", "Chongqing, China", "Haerbin, China", "Chengdu, China", "Xian, China", "Nanjing, China", "Taipei, China", "Zibo, China", "Dalian, China", "Jinan, China", "Changchun, China", "Qingdao, China", "Taiyuan, China"],
    "Japan": ["Tokyo, Japan", "Yokohama, Japan", "Osaka, Japan", "Nagoya, Japan"],
    "USA": ["New York, USA", "Los Angeles, USA", "Chicago, USA", "Houston, USA"],
    "Thailand": ["Bangkok, Thailand"],
    "UK": ["London, UK"],
    "Egypt": ["Cairo, Egypt", "Alexandria, Egypt", "Giza, Egypt"],
    "Iran": ["Tehran, Iran", "Mashhad, Iran"],
    "Colombia": ["Bogota, Colombia", "Cali, Colombia", "Medellin, Colombia"],
    "Peru": ["Lima, Peru"],
    "Chile": ["Santiago, Chile"],
    "Australia": ["Sydney, Australia"],
    "Singapore": ["Singapore, Singapore"],
    "Iraq": ["Baghdad, Iraq"],
    "Bangladesh": ["Dhaka, Bangladesh"],
    "Germany": ["Berlin, Germany"],
    "Vietnam": ["Ho Chi Minh City, Vietnam"],
    "Argentina": ["Buenos Aires, Argentina"],
    "Morocco": ["Casablanca, Morocco"],
    "Spain": ["Madrid, Spain"],
    "North Korea": ["Pyong Yang, North Korea"],
    "Congo": ["Kinshaha, Congo"],
    "Italy": ["Rome, Italy"],
    "Ukraine": ["Kiev, Ukraine"],
    "Myanmar": ["Yangon, Myanmar"],
    "Canada": ["Toronto, Canada"],
    "Ethiopia": ["Addis Ababa, Ethiopia"],
    "Cuba": ["Havanna, Cuba"],
    "France": ["Paris, France"],
    "Uzbekistan": ["Tashkent, Uzbekistan"],
    "Ecuador": ["Guayaquil, Ecuador"],
    "Romania": ["Bucuresti, Romania"],
    "Philippines": ["Quezon City, Philippines"],
    "Cote d'Ivorie": ["Abidjan, Cote d'Ivorie"],
    "Hungary": ["Budapest, Hungary"],
    "Venezuela": ["Caracas, Venezuela"]
}

def get_default_word_list(data_split: str = "all") -> List[WordVariants]:
    if data_split == "all":
        dict_to_use = DEFAULT_CITY_DICT
    elif data_split == "train":
        dict_to_use = DEFAULT_CITY_DICT
    elif data_split == "val":
        dict_to_use = DEFAULT_CITY_DICT
    elif data_split == "test":
        dict_to_use = DEFAULT_CITY_DICT
    else:
        raise ValueError(f"Invalid data split: {data_split}")
    
    word_list = []
    for _, words in dict_to_use.items():
        word_list.extend(map(lambda x: WordVariants.from_str(x), words))
    return word_list

def is_done(city: WordVariants, question: str):
    # Use just the raw city name before the comma
    city_name = city.words[0].split(",")[0].lower().strip()

    question = question.rstrip("?.!").lower().strip()

    guess_patterns = [
        rf"is it {city_name}",
        rf"is the city {city_name}",
        rf"is it the city of {city_name}",
        rf"is the city called {city_name}",
        rf"is the place {city_name}",
        rf"is the place called {city_name}",
        rf"are you from {city_name}"
    ]

    if question == city_name:
        return True

    for pattern in guess_patterns:
        if re.fullmatch(pattern, question):
            return True

    tokens = nltk.word_tokenize(question)
    if tokens and tokens[-1] == city_name:
        return True

    return False
