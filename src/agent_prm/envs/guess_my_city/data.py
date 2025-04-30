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
        "Delhi, India;New Delhi, India",
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
        "Xian, China;Xi'an, China",
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
        "Chiang Mai, Thailand;Chiengmai, Thailand;Chiangmai, Thailand",
        "Pattaya, Thailand;Phatthaya, Thailand;Pattayá, Thailand",
        "Tehran, Iran",
        "Mashhad, Iran",
        "Isfahan, Iran;Esfahan, Iran;Ispahan, Iran",
        "Singapore, Singapore",
        "Sentosa, Singapore;Pulau Blakang Mati, Singapore;Sentosa Island, Singapore",
        "Jurong, Singapore;Jurong East, Singapore;Jurong West, Singapore",
        "Baghdad, Iraq",
        "Mosul, Iraq;Al-Mawsil, Iraq;Mawsil, Iraq;Nineveh, Iraq",
        "Erbil, Iraq;Arbil, Iraq;Hawler, Iraq;Hewlêr, Iraq",
        "Dhaka, Bangladesh",
        "Chittagong, Bangladesh;Chattogram, Bangladesh;Chattagaon, Bangladesh",
        "Khulna, Bangladesh;Kholna, Bangladesh;Kulna, Bangladesh",
        "Ho Chi Minh City, Vietnam;Saigon, Vietnam",
        "Hanoi, Vietnam;Hà Nôi, Vietnam;Ha Noi, Vietnam;Thǎng Long, Vietnam",
        "Da Nang, Vietnam;Ðà Nǎng, Vietnam;Danang, Vietnam;Tourane, Vietnam",
        "Pyong Yang, North Korea;Pyongyang, North Korea",
        "Hamhung, North Korea;Hamhǔng, North Korea;Ham Heung, North Korea;Hangul, North Korea",
        "Chongjin, North Korea;Chǒngjin, North Korea;Cheongjin, North Korea;Hangul, North Korea",
        "Yangon, Myanmar",
        "Mandalay, Myanmar;Mandalay City, Myanmar",
        "Naypyidaw, Myanmar;Nay Pyi Taw, Myanmar;Naypyitaw, Myanmar",
        "Tashkent, Uzbekistan",
        "Samarkand, Uzbekistan;Samarqand, Uzbekistan",
        "Bukhara, Uzbekistan;Buxoro, Uzbekistan",
        "Quezon City, Philippines;QC, Philippines",
        "Manila, Philippines;Maynila, Philippines",
        "Cebu City, Philippines;Cebu, Philippines"
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
        "Cusco, Peru;Cuzco, Peru;Qosqo, Peru",
        "Arequipa, Peru;Areqipa, Peru",
        "Santiago, Chile",
        "Valparaíso, Chile;Valparaiso, Chile;Valpo, Chile",
        "Concepción, Chile;Concepcion, Chile;Conce, Chile",
        "Buenos Aires, Argentina",
        "Córdoba, Argentina;Cordoba, Argentina",
        "Rosario, Argentina;Rosario City, Argentina",
        "Guayaquil, Ecuador",
        "Quito, Ecuador;San Francisco de Quito, Ecuador",
        "Cuenca, Ecuador;Santa Ana de los Ríos de Cuenca, Ecuador;Cuenca City, Ecuador",
        "Caracas, Venezuela",
        "Maracaibo, Venezuela;Maracaibo City, Venezuela",
        "Valencia, Venezuela;Valencia del Rey, Venezuela;Valencia City, Venezuela"
    ],
    "Europe": [
        "Moscow, Russia",
        "St Petersburg, Russia;St. Petersburg, Russia",
        "Novosibirsk, Russia;Novosibirsk City, Russia",
        "London, UK",
        "Birmingham, UK;Brum, UK;B'ham, UK",
        "Edinburgh, UK;Edinborough, UK",
        "Berlin, Germany",
        "Munich, Germany;München, Germany",
        "Cologne, Germany;Köln, Germany",
        "Madrid, Spain",
        "Barcelona, Spain;Barna, Spain",
        "Valencia, Spain",
        "Rome, Italy",
        "Naples, Italy;Napoli, Italy",
        "Milan, Italy;Milano, Italy",
        "Kiev, Ukraine;Kyiv, Ukraine",
        "Odesa, Ukraine;Odessa, Ukraine",
        "Dnipro, Ukraine;Dnipropetrovsk, Ukraine;Dnepropetrovsk, Ukraine",
        "Paris, France",
        "Marseille, France;Marseilles, France;Massalia, France",
        "Lyon, France;Lyons, France;Lugdunum, France",
        "Bucuresti, Romania;Bucharest, Romania",
        "Cluj-Napoca, Romania;Cluj, Romania;Cluj Napoca, Romania",
        "Timișoara, Romania;Timisoara, Romania",
        "Budapest, Hungary",
        "Debrecen, Hungary;Debretin, Hungary",
        "Szeged, Hungary"
    ],
    "North America": [
        "Mexico City, Mexico",
        "Guadalajara, Mexico;Guad, Mexico",
        "Monterrey, Mexico;Monterrei, Mexico",
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "Toronto, Canada",
        "Vancouver, Canada",
        "Montreal, Canada;Ville-Marie, Canada",
        "Havanna, Cuba;Havana, Cuba",
        "Santiago de Cuba, Cuba;Santiago, Cuba",
        "Camagüey, Cuba;Puerto Príncipe, Cuba;Camaguey, Cuba"
    ],
    "Africa": [
        "Cairo, Egypt",
        "Alexandria, Egypt",
        "Giza, Egypt",
        "Casablanca, Morocco",
        "Marrakesh, Morocco;Marrakech, Morocco",
        "Fes, Morocco;Fez, Morocco",
        "Kinshaha, Congo;Kinshasa, Congo",
        "Lubumbashi, Congo;Élisabethville, Congo;Lumbumbashi, Congo",
        "Mbuji-Mayi, Congo;Mbujimayi, Congo;Mbuji Mayi, Congo;Bakwanga, Congo",
        "Addis Ababa, Ethiopia",
        "Dire Dawa, Ethiopia;Dire Dhawa, Ethiopia;Diré-Daoua, Ethiopia;Dir Dawa, Ethiopia",
        "Gondar, Ethiopia;Gonder, Ethiopia;Gondär, Ethiopia;Gondaré, Ethiopia",
        "Abidjan, Cote d'Ivorie;Abidjan, Côte d'Ivoire",
        "Yamoussoukro, Cote d'Ivorie;Yamoussukro, Cote d'Ivorie;Yamassoukro, Cote d'Ivorie;Yamousouko, Cote d'Ivorie",
        "Bouaké, Cote d'Ivorie;Bouake, Cote d'Ivorie;Bwake, Cote d'Ivorie;Bouaké City, Cote d'Ivorie"
    ],
    "Australia": [
        "Sydney, Australia",
        "Melbourne, Australia;Melburn, Australia;Melbourne City, Australia",
        "Brisbane, Australia;Brisbane City, Australia"
    ]
}

TRAIN_CITY_DICT = {
    "Asia": [
        "Busan, South Korea;Pusan, South Korea",
        "Daegu, South Korea;Taegu, South Korea;Taega, South Korea",
        "Incheon, South Korea;Inchon, South Korea",
        "Delhi, India;New Delhi, India",
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
        "Xian, China;Xi'an, China",
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
        "Chiang Mai, Thailand;Chiengmai, Thailand;Chiangmai, Thailand",
        "Pattaya, Thailand;Phatthaya, Thailand;Pattayá, Thailand",
        "Mashhad, Iran",
        "Isfahan, Iran;Esfahan, Iran;Ispahan, Iran",
        "Sentosa, Singapore;Pulau Blakang Mati, Singapore;Sentosa Island, Singapore",
        "Jurong, Singapore;Jurong East, Singapore;Jurong West, Singapore",
        "Mosul, Iraq;Al-Mawsil, Iraq;Mawsil, Iraq;Nineveh, Iraq",
        "Erbil, Iraq;Arbil, Iraq;Hawler, Iraq;Hewlêr, Iraq",
        "Chittagong, Bangladesh;Chattogram, Bangladesh;Chattagaon, Bangladesh",
        "Khulna, Bangladesh;Kholna, Bangladesh;Kulna, Bangladesh",
        "Hanoi, Vietnam;Hà Nôi, Vietnam;Ha Noi, Vietnam;Thǎng Long, Vietnam",
        "Da Nang, Vietnam;Ðà Nǎng, Vietnam;Danang, Vietnam;Tourane, Vietnam",
        "Hamhung, North Korea;Hamhǔng, North Korea;Ham Heung, North Korea;Hangul, North Korea",
        "Chongjin, North Korea;Chǒngjin, North Korea;Cheongjin, North Korea;Hangul, North Korea",
        "Mandalay, Myanmar;Mandalay City, Myanmar",
        "Naypyidaw, Myanmar;Nay Pyi Taw, Myanmar;Naypyitaw, Myanmar",
        "Samarkand, Uzbekistan;Samarqand, Uzbekistan",
        "Bukhara, Uzbekistan;Buxoro, Uzbekistan",
        "Manila, Philippines;Maynila, Philippines",
        "Cebu City, Philippines;Cebu, Philippines"
    ],
    "South America": [
        "Rio de Janeiro, Brazil",
        "Salvador, Brazil",
        "Belo Horizonte, Brazil",
        "Fortaleza, Brazil",
        "Brasilia, Brazil",
        "Cali, Colombia",
        "Medellin, Colombia",
        "Cusco, Peru;Cuzco, Peru;Qosqo, Peru",
        "Arequipa, Peru;Areqipa, Peru",
        "Valparaíso, Chile;Valparaiso, Chile;Valpo, Chile",
        "Concepción, Chile;Concepcion, Chile;Conce, Chile",
        "Córdoba, Argentina;Cordoba, Argentina",
        "Rosario, Argentina;Rosario City, Argentina",
        "Quito, Ecuador;San Francisco de Quito, Ecuador",
        "Cuenca, Ecuador;Santa Ana de los Ríos de Cuenca, Ecuador;Cuenca City, Ecuador",
        "Maracaibo, Venezuela;Maracaibo City, Venezuela",
        "Valencia, Venezuela;Valencia del Rey, Venezuela;Valencia City, Venezuela"
    ],
    "Europe": [
        "St Petersburg, Russia;St. Petersburg, Russia",
        "Novosibirsk, Russia;Novosibirsk City, Russia",
        "Birmingham, UK;Brum, UK;B'ham, UK",
        "Edinburgh, UK;Edinborough, UK",
        "Munich, Germany;München, Germany",
        "Cologne, Germany;Köln, Germany",
        "Barcelona, Spain;Barna, Spain",
        "Valencia, Spain",
        "Naples, Italy;Napoli, Italy",
        "Milan, Italy;Milano, Italy",
        "Odesa, Ukraine;Odessa, Ukraine",
        "Dnipro, Ukraine;Dnipropetrovsk, Ukraine;Dnepropetrovsk, Ukraine",
        "Marseille, France;Marseilles, France;Massalia, France",
        "Lyon, France;Lyons, France;Lugdunum, France",
        "Cluj-Napoca, Romania;Cluj, Romania;Cluj Napoca, Romania",
        "Timișoara, Romania;Timisoara, Romania",
        "Debrecen, Hungary;Debretin, Hungary",
        "Szeged, Hungary"
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
        "Bangkok, Thailand",
        "Tehran, Iran",
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
        "Bogota, Colombia;Bogotá, Colombia",
        "Lima, Peru",
        "Santiago, Chile",
        "Buenos Aires, Argentina",
        "Guayaquil, Ecuador",
        "Caracas, Venezuela"
    ],
    "Europe": [
        "Moscow, Russia",
        "London, UK",
        "Berlin, Germany",
        "Madrid, Spain",
        "Rome, Italy",
        "Kiev, Ukraine;Kyiv, Ukraine",
        "Paris, France",
        "Bucuresti, Romania;Bucharest, Romania",
        "Budapest, Hungary",
    ]
}

TEST_CITY_DICT = {
    "North America": [
        "Mexico City, Mexico",
        "Guadalajara, Mexico;Guad, Mexico",
        "Monterrey, Mexico;Monterrei, Mexico",
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "Toronto, Canada",
        "Vancouver, Canada",
        "Montreal, Canada;Ville-Marie, Canada",
        "Havanna, Cuba;Havana, Cuba",
        "Santiago de Cuba, Cuba;Santiago, Cuba",
        "Camagüey, Cuba;Puerto Príncipe, Cuba;Camaguey, Cuba"
    ],
    "Africa": [
        "Cairo, Egypt",
        "Alexandria, Egypt",
        "Giza, Egypt",
        "Casablanca, Morocco",
        "Marrakesh, Morocco;Marrakech, Morocco",
        "Fes, Morocco;Fez, Morocco",
        "Kinshaha, Congo;Kinshasa, Congo",
        "Lubumbashi, Congo;Élisabethville, Congo;Lumbumbashi, Congo",
        "Mbuji-Mayi, Congo;Mbujimayi, Congo;Mbuji Mayi, Congo;Bakwanga, Congo",
        "Addis Ababa, Ethiopia",
        "Dire Dawa, Ethiopia;Dire Dhawa, Ethiopia;Diré-Daoua, Ethiopia;Dir Dawa, Ethiopia",
        "Gondar, Ethiopia;Gonder, Ethiopia;Gondär, Ethiopia;Gondaré, Ethiopia",
        "Abidjan, Cote d'Ivorie;Abidjan, Côte d'Ivoire",
        "Yamoussoukro, Cote d'Ivorie;Yamoussukro, Cote d'Ivorie;Yamassoukro, Cote d'Ivorie;Yamousouko, Cote d'Ivorie",
        "Bouaké, Cote d'Ivorie;Bouake, Cote d'Ivorie;Bwake, Cote d'Ivorie;Bouaké City, Cote d'Ivorie"
    ],
    "Australia": [
        "Sydney, Australia",
        "Melbourne, Australia;Melburn, Australia;Melbourne City, Australia",
        "Brisbane, Australia;Brisbane City, Australia"
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

    return False
