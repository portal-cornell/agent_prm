"""
Adapted from https://github.com/abdulhaim/LMRL-Gym
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import nltk
from collections import defaultdict
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
    "Asia": [
        "Seoul, South Korea",
        "Busan, South Korea;Pusan, South Korea",
        "Daegu, South Korea;Taegu, South Korea;Taega, South Korea",
        "Incheon, South Korea;Inchon, South Korea",
        "Bombay, India;Mumbai, India",
        "Delhi, India",
        "Calcutta, India;Kolkata, India",
        "Madras, India;Chennai, India",
        "Bangalore, India;Bengaluru, India",
        "Hyderabad, India",
        "Ahmedabad, India",
        "Kanpur, India",
        "Jakarta, Indonesia;Djakarta, Indonesia",
        "Bandung, Indonesia",
        "Bagor, Indonesia",
        "Malang, Indonesia",
        "Surabaya, Indonesia",
        "Semarang, Indonesia",
        "Sukabumi, Indonesia",
        "Cirebon, Indonesia",
        "Medan, Indonesia",
        "Karachi, Pakistan",
        "Lahore, Pakistan",
        "Faisalabad, Pakistan",
        "Istanbul, Turkey",
        "Ankara, Turkey",
        "Izmir, Turkey",
        "Shanghai, China",
        "Beijing, China",
        "Hong Kong, China",
        "Tianjin, China",
        "Shenyang, China",
        "Wuhan, China",
        "Guangzhou, China",
        "Chongqing, China",
        "Haerbin, China",
        "Chengdu, China",
        "Xian, China",
        "Nanjing, China",
        "Taipei, China",
        "Zibo, China",
        "Dalian, China",
        "Jinan, China",
        "Changchun, China",
        "Qingdao, China",
        "Taiyuan, China",
        "Tokyo, Japan",
        "Yokohama, Japan",
        "Osaka, Japan",
        "Nagoya, Japan",
        "Bangkok, Thailand",
        "Tehran, Iran",
        "Mashhad, Iran",
        "Singapore, Singapore",
        "Baghdad, Iraq",
        "Dhaka, Bangladesh",
        "Ho Chi Minh City, Vietnam;Saigon, Vietnam",
        "Pyong Yang, North Korea;Pyongyang, North Korea",
        "Yangon, Myanmar",
        "Tashkent, Uzbekistan",
        "Quezon City, Philippines;QC, Philippines"
    ],
    "South America": [
        "Sao Paulo, Brazil;São Paulo, Brazil",
        "Rio de Janeiro, Brazil",
        "Salvador, Brazil",
        "Belo Horizonte, Brazil",
        "Fortaleza, Brazil",
        "Brasilia, Brazil",
        "Bogota, Colombia;Bogotá, Colombia",
        "Cali, Colombia",
        "Medellin, Colombia",
        "Lima, Peru",
        "Santiago, Chile",
        "Buenos Aires, Argentina",
        "Guayaquil, Ecuador",
        "Caracas, Venezuela"
    ],
    "Europe": [
        "Moscow, Russia",
        "St Petersburg, Russia",
        "Istanbul, Turkey",
        "London, UK",
        "Berlin, Germany",
        "Madrid, Spain",
        "Rome, Italy",
        "Kiev, Ukraine;Kyiv, Ukraine",
        "Paris, France",
        "Bucuresti, Romania;Bucharest, Romania",
        "Budapest, Hungary"
    ],
    "North America": [
        "Mexico City, Mexico",
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "Toronto, Canada",
        "Havanna, Cuba;Havana, Cuba"
    ],
    "Africa": [
        "Cairo, Egypt",
        "Alexandria, Egypt",
        "Giza, Egypt",
        "Casablanca, Morocco",
        "Kinshaha, Congo;Kinshasa, Congo",
        "Addis Ababa, Ethiopia",
        "Abidjan, Cote d'Ivorie;Abidjan, Côte d'Ivoire"
    ],
    "Australia": [
        "Sydney, Australia"
    ]
}

TRAIN_CITY_DICT = {
    "Asia": [
        "Busan, South Korea;Pusan, South Korea",
        "Daegu, South Korea;Taegu, South Korea;Taega, South Korea",
        "Incheon, South Korea;Inchon, South Korea",
        "Delhi, India",
        "Calcutta, India;Kolkata, India",
        "Madras, India;Chennai, India",
        "Bangalore, India;Bengaluru, India",
        "Hyderabad, India",
        "Ahmedabad, India",
        "Kanpur, India",
        "Bandung, Indonesia",
        "Bagor, Indonesia",
        "Malang, Indonesia",
        "Surabaya, Indonesia",
        "Semarang, Indonesia",
        "Sukabumi, Indonesia",
        "Cirebon, Indonesia",
        "Medan, Indonesia",
        "Lahore, Pakistan",
        "Faisalabad, Pakistan",
        "Ankara, Turkey",
        "Izmir, Turkey",
        "Beijing, China",
        "Hong Kong, China",
        "Tianjin, China",
        "Shenyang, China",
        "Wuhan, China",
        "Guangzhou, China",
        "Chongqing, China",
        "Haerbin, China",
        "Chengdu, China",
        "Xian, China",
        "Nanjing, China",
        "Taipei, China",
        "Zibo, China",
        "Dalian, China",
        "Jinan, China",
        "Changchun, China",
        "Qingdao, China",
        "Taiyuan, China",
        "Yokohama, Japan",
        "Osaka, Japan",
        "Nagoya, Japan",
        "Bangkok, Thailand",
        "Mashhad, Iran",
        "Singapore, Singapore",
        "Baghdad, Iraq",
        "Dhaka, Bangladesh",
        "Ho Chi Minh City, Vietnam;Saigon, Vietnam",
        "Pyong Yang, North Korea;Pyongyang, North Korea",
        "Yangon, Myanmar",
        "Tashkent, Uzbekistan",
        "Quezon City, Philippines;QC, Philippines"
    ],
    "South America": [
        "Rio de Janeiro, Brazil",
        "Salvador, Brazil",
        "Belo Horizonte, Brazil",
        "Fortaleza, Brazil",
        "Brasilia, Brazil",
        "Cali, Colombia",
        "Medellin, Colombia",
        "Lima, Peru",
        "Santiago, Chile",
        "Buenos Aires, Argentina",
        "Guayaquil, Ecuador",
        "Caracas, Venezuela"
    ],
    "Europe": [
        "St Petersburg, Russia",
        "Istanbul, Turkey",
        "London, UK",
        "Berlin, Germany",
        "Madrid, Spain",
        "Rome, Italy",
        "Kiev, Ukraine;Kyiv, Ukraine",
        "Paris, France",
        "Bucuresti, Romania;Bucharest, Romania",
        "Budapest, Hungary"
    ]
}

VALIDATION_CITY_DICT = {
    "Asia": [
        "Seoul, South Korea",
        "Bombay, India;Mumbai, India",
        "Jakarta, Indonesia;Djakarta, Indonesia",
        "Karachi, Pakistan",
        "Istanbul, Turkey",
        "Shanghai, China",
        "Tokyo, Japan",
        "Tehran, Iran",
    ],
    "South America": [
        "Sao Paulo, Brazil;São Paulo, Brazil",
        "Bogota, Colombia;Bogotá, Colombia"
    ],
    "Europe": [
        "Moscow, Russia"
    ]
}

TEST_CITY_DICT = {
    "North America": [
        "Mexico City, Mexico",
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "Toronto, Canada",
        "Havanna, Cuba;Havana, Cuba"
    ],
    "Africa": [
        "Cairo, Egypt",
        "Alexandria, Egypt",
        "Giza, Egypt",
        "Casablanca, Morocco",
        "Kinshaha, Congo;Kinshasa, Congo",
        "Addis Ababa, Ethiopia",
        "Abidjan, Cote d'Ivorie;Abidjan, Côte d'Ivoire"
    ],
    "Australia": [
        "Sydney, Australia"
    ]
}

def get_default_word_list(data_split: str = "all") -> List[WordVariants]:
    if data_split == "all":
        dict_to_use = DEFAULT_CITY_DICT
    elif data_split == "train":
        dict_to_use = TRAIN_CITY_DICT
    elif data_split == "val":
        dict_to_use = VALIDATION_CITY_DICT
    elif data_split == "test":
        dict_to_use = TEST_CITY_DICT
    else:
        raise ValueError(f"Invalid data split: {data_split}")
    
    word_list = []
    for _, words in dict_to_use.items():
        word_list.extend(map(lambda x: WordVariants.from_str(x), words))
    return word_list

def is_done(city: WordVariants, question: str):
    question = question.rstrip("?.!").lower().strip()

    city_names = [
        w.split(",")[0].strip().lower()
        for w in city.words
    ]

    if question in city_names:
        return True

    for city_name in city_names:
        guess_patterns = [
            rf"is it {city_name}",
            rf"is the city {city_name}",
            rf"is it the city of {city_name}",
            rf"is the city called {city_name}",
            rf"is the place {city_name}",
            rf"is the place called {city_name}",
            rf"are you from {city_name}",
            rf"is the city .*{city_name}.*", 
        ]

        for pattern in guess_patterns:
            if re.fullmatch(pattern, question):
                return True

    tokens = nltk.word_tokenize(question)
    if tokens and tokens[-1] in city_names:
        return True

    return False
