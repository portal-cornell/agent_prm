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
        "Daejeon, South Korea;Taejon, South Korea;Daejeon-si, South Korea",
        "Gwangju, South Korea;Kwangju, South Korea;Gwangju Metropolitan City, South Korea",
        "Suwon, South Korea;Suwon-si, South Korea",
        "Ulsan, South Korea;Ulsan Metropolitan City, South Korea",
        "Jeonju, South Korea;Chǒnju, South Korea;Jeonju-si, South Korea",
        "Gyeongju, South Korea;Kyǒngju, South Korea;Geyongju-si, South Korea;Seorabeol, South Korea",
        "Bombay, India;Mumbai, India",
        "Delhi, India;New Delhi, India",
        "Calcutta, India;Kolkata, India",
        "Madras, India;Chennai, India",
        "Bangalore, India;Bengaluru, India",
        "Hyderabad, India",
        "Ahmedabad, India",
        "Kanpur, India",
        "Bengaluru, India;Bangalore, India;Bengalooru, India",
        "Chennai, India;Madras, India",
        "Jakarta, Indonesia;Djakarta, Indonesia",
        "Bandung, Indonesia",
        "Bagor, Indonesia",
        "Malang, Indonesia",
        "Surabaya, Indonesia",
        "Semarang, Indonesia",
        "Sukabumi, Indonesia",
        "Cirebon, Indonesia",
        "Medan, Indonesia",
        "Yogyakarta, Indonesia;Jogjakarta, Indonesia;Jogja, Indonesia;Kota Yogyakarta, Indonesia",
        "Manila, Philippines",
        "Quezon City, Philippines;Quezon, Philippines",
        "Cebu City, Philippines;Cebu, Philippines",
        "Davao City, Philippines;Davao, Philippines",
        "Zamboanga City, Philippines;Zamboanga, Philippines",
        "Baguio, Philippines;Baguio City, Philippines",
        "Iloilo City, Philippines;Iloilo, Philippines",
        "Cagayan de Oro, Philippines;Cagayan, Philippines",
        "General Santos, Philippines;General Santos City, Philippines",
        "Taguig, Philippines;Taguig City, Philippines",
        "Shanghai, China",
        "Beijing, China",
        "Hong Kong, China",
        "Tianjin, China",
        "Shenyang, China",
        "Wuhan, China",
        "Guangzhou, China",
        "Chongqing, China",
        "Haerbin, China;Harbin, China",
        "Chengdu, China"
    ],
    "South America": [
        "Sao Paulo, Brazil;São Paulo, Brazil",
        "Rio de Janeiro, Brazil",
        "Salvador, Brazil",
        "Belo Horizonte, Brazil",
        "Fortaleza, Brazil",
        "Brasilia, Brazil",
        "Recife, Brazil;Recife City, Brazil",
        "Curitiba, Brazil;Curita, Brazil;Curitiba City, Brazil",
        "Manaus, Brazil;Manaós, Brazil;Manaus City, Brazil",
        "Porto Alegre, Brazil;Porto, Brazil",
        "Santiago, Chile",
        "Valparaíso, Chile;Valparaiso, Chile;Valpo, Chile",
        "Concepción, Chile;Concepcion, Chile;Conce, Chile",
        "La Serena, Chile;Serena, Chile;La Serena City, Chile",
        "Antofagasta, Chile;Antofa, Chile;Antofagasta City, Chile",
        "Temuco, Chile;Temuco City, Chile",
        "Puerto Montt, Chile;Puerto, Chile",
        "Iquique, Chile;Iquique City, Chile",
        "Arica, Chile;Arica City, Chile",
        "Rancagua, Chile;Rancagua City, Chile",
        "Buenos Aires, Argentina",
        "Córdoba, Argentina;Cordoba, Argentina",
        "Rosario, Argentina;Rosario City, Argentina",
        "Mendoza, Argentina;Mendoza City, Argentina",
        "La Plata, Argentina;La Plata City, Argentina",
        "San Miguel de Tucumán, Argentina;Tucumán, Argentina",
        "Mar del Plata, Argentina;Mardel, Argentina;Mar del Plata City, Argentina",
        "Salta, Argentina;Salta City, Argentina",
        "San Juan, Argentina;San Juan City, Argentina",
        "Neuquén, Argentina;Neuquen, Argentina;Neuquén Capital, Argentina",
        "Guayaquil, Ecuador",
        "Quito, Ecuador;San Francisco de Quito, Ecuador",
        "Cuenca, Ecuador;Santa Ana de los Ríos de Cuenca, Ecuador;Cuenca City, Ecuador",
        "Ambato, Ecuador;Ambato City, Ecuador",
        "Machala, Ecuador;Machala City, Ecuador",
        "Loja, Ecuador;Loja City, Ecuador",
        "Esmeraldas, Ecuador;Esmeraldas City, Ecuador",
        "Riobamba, Ecuador;Riobamba City, Ecuador",
        "Manta, Ecuador;Manta City, Ecuador;Puerto de Manta, Ecuador",
        "Portoviejo, Ecuador;Portoviejo City, Ecuador",
        "Caracas, Venezuela",
        "Maracaibo, Venezuela;Maracaibo City, Venezuela",
        "Valencia, Venezuela;Valencia del Rey, Venezuela;Valencia City, Venezuela",
        "Barquisimeto, Venezuela;Barquisimeto City, Venezuela",
        "Mérida, Venezuela;Merida, Venezuela",
        "Puerto La Cruz, Venezuela;Puerto, Venezuela;La Cruz, Venezuela",
        "Ciudad Guayana, Venezuela;Cuidad Guayana City, Venezuela",
        "San Cristóbal, Venezuela;San Cristobal, Venezuela",
        "Barcelona, Venezuela;Barcelona City, Venezuela;Barcelona de Venezuela, Venezuela",
        "Cumaná, Venezuela;Cumana, Venezuela"
    ],
    "Europe": [
        "Rome, Italy",
        "Naples, Italy;Napoli, Italy",
        "Milan, Italy;Milano, Italy",
        "Florence, Italy;Firenze, Italy",
        "Venice, Italy;Venezia, Italy",
        "Turin, Italy;Torina, Italy",
        "Bologna, Italy;Bologna City, Italy",
        "Genoa, Italy;Genova, Italy",
        "Palermo, Italy;Panormus, Italy;Palermo City, Italy",
        "Verona, Italy;Verona City, Italy",
        "Kiev, Ukraine;Kyiv, Ukraine",
        "Odesa, Ukraine;Odessa, Ukraine",
        "Dnipro, Ukraine;Dnipropetrovsk, Ukraine;Dnepropetrovsk, Ukraine",
        "Lviv, Ukraine;Lvov, Ukraine",
        "Kharkiv, Ukraine;Kharkov, Ukraine",
        "Zaporizhzhia, Ukraine;Zaporozhye, Ukraine",
        "Vinnytsia, Ukraine;Vinnitsa, Ukraine;Vinnytsia City, Ukraine",
        "Mykolaiv, Ukraine;Nikolaev, Ukraine;Mykolaiv City, Ukraine",
        "Chernihiv, Ukraine;Chernigov, Ukraine;Chernihiv City, Ukraine",
        "Ivano-Frankivsk, Ukraine;Stanislav, Ukraine",
        "London, UK",
        "Manchester, UK",
        "Birmingham, UK",
        "Glasgow, UK",
        "Edinburgh, UK",
        "Liverpool, UK",
        "Leeds, UK;Leeds City, UK",
        "Bristol, UK;Bristol City, UK",
        "Sheffield, UK;Sheffield City, UK",
        "Cardiff, UK",
        "Bucuresti, Romania;Bucharest, Romania",
        "Cluj-Napoca, Romania;Cluj, Romania;Cluj Napoca, Romania",
        "Timișoara, Romania;Timisoara, Romania",
        "Iași, Romania;Iasi, Romania;Jassy, Romania",
        "Constanța, Romania;Constanta, Romania",
        "Brașov, Romania;Brasov, Romania",
        "Sibiu, Romania;Sibiu City, Romania",
        "Oradea, Romania;Oradea City, Romania",
        "Galați, Romania;Galati, Romania;Galați City, Romania",
        "Ploiești, Romania;Ploiesti, Romania;Ploești, Romania;Ploiești City, Romania",
        "Budapest, Hungary",
        "Debrecen, Hungary;Debretin, Hungary",
        "Szeged, Hungary",
        "Miskolc, Hungary;Miskolcz, Hungary;Miskolc City, Hungary",
        "Pécs, Hungary;Pecs, Hungary",
        "Gyor, Hungary;Gyõr, Hungary",
        "Nyíregyháza, Hungary;Nyiregyhaza, Hungary;Nyíregyháza City, Hungary",
        "Eger, Hungary;Eger City, Hungary",
        "Székesfehérvár, Hungary;Szekesfehervar, Hungary;Szfvár, Hungary",
        "Kecskemét, Hungary;Kecskemet, Hungary;Kecskemét City, Hungary"
    ],
    "North America": [
        "Toronto, Canada",
        "Vancouver, Canada",
        "Montreal, Canada;Ville-Marie, Canada",
        "Calgary, Canada;Calgary City, Canada",
        "Ottawa, Canada;Ottawa-Gatineau, Canada",
        "Edmonton, Canada;Edmonton City, Canada",
        "Quebec City, Canada;Québec, Canada;Ville de Québec, Canada",
        "Winnipeg, Canada",
        "Hamilton, Canada;Hamilton City, Canada",
        "Halifax, Canada;Halifax Regional Municipality, Canada",
        "Havanna, Cuba;Havana, Cuba",
        "Santiago de Cuba, Cuba;Santiago, Cuba",
        "Camagüey, Cuba;Puerto Príncipe, Cuba;Camaguey, Cuba",
        "Holguín, Cuba;Holguin, Cuba",
        "Santa Clara, Cuba;Santa Clara City, Cuba",
        "Guantánamo, Cuba;Guantanamo, Cuba;Guantánamo City, Cuba",
        "Cienfuegos, Cuba;Cienfuegos City, Cuba",
        "Bayamo, Cuba;Bayamo City, Cuba",
        "Mantanzas, Cuba;Mantanzas City, Cuba",
        "Pinar del Río, Cuba;Pinar del Rio, Cuba;Pinar City, Cuba"
    ],
    "Africa": [
        "Addis Ababa, Ethiopia",
        "Dire Dawa, Ethiopia;Dire Dhawa, Ethiopia;Diré-Daoua, Ethiopia;Dir Dawa, Ethiopia",
        "Gondar, Ethiopia;Gonder, Ethiopia;Gondär, Ethiopia;Gondaré, Ethiopia",
        "Mekelle, Ethiopia;Mekele, Ethiopia;Mäqälle, Ethiopia",
        "Bahir Dar, Ethiopia;Bahar Dar, Ethiopia;Bahirdar, Ethiopia",
        "Hawassa, Ethiopia;Awassa, Ethiopia;Hawasa, Ethiopia",
        "Jimma, Ethiopia;Jima, Ethiopia;Jimma City, Ethiopia",
        "Harar, Ethiopia;Harer, Ethiopia;Adare, Ethiopia",
        "Dessie, Ethiopia;Dessye, Ethiopia",
        "Adama, Ethiopia;Nazreth, Ethiopia;Nazret, Ethiopia"
    ]
}

TRAIN_CITY_DICT = {
    "Asia": [
        "Busan, South Korea;Pusan, South Korea",
        "Incheon, South Korea;Inchon, South Korea",
        "Daejeon, South Korea;Taejon, South Korea;Daejeon-si, South Korea",
        "Gwangju, South Korea;Kwangju, South Korea;Gwangju Metropolitan City, South Korea",
        "Suwon, South Korea;Suwon-si, South Korea",
        "Ulsan, South Korea;Ulsan Metropolitan City, South Korea",
        "Jeonju, South Korea;Chǒnju, South Korea;Jeonju-si, South Korea",
        "Gyeongju, South Korea;Kyǒngju, South Korea;Geyongju-si, South Korea;Seorabeol, South Korea",
        "Calcutta, India;Kolkata, India",
        "Madras, India;Chennai, India",
        "Hyderabad, India",
        "Ahmedabad, India",
        "Kanpur, India",
        "Bombay, India;Mumbai, India",
        "Bengaluru, India;Bangalore, India;Bengalooru, India",
        "Chennai, India;Madras, India",
        "Bandung, Indonesia",
        "Malang, Indonesia",
        "Surabaya, Indonesia",
        "Semarang, Indonesia",
        "Sukabumi, Indonesia",
        "Cirebon, Indonesia",
        "Medan, Indonesia",
        "Yogyakarta, Indonesia;Jogjakarta, Indonesia;Jogja, Indonesia;Kota Yogyakarta, Indonesia",
        "Cebu City, Philippines;Cebu, Philippines",
        "Davao City, Philippines;Davao, Philippines",
        "Zamboanga City, Philippines;Zamboanga, Philippines",
        "Baguio, Philippines;Baguio City, Philippines",
        "Iloilo City, Philippines;Iloilo, Philippines",
        "Cagayan de Oro, Philippines;Cagayan, Philippines",
        "General Santos, Philippines;General Santos City, Philippines",
        "Taguig, Philippines;Taguig City, Philippines",
        "Hong Kong, China",
        "Tianjin, China",
        "Shenyang, China",
        "Wuhan, China",
        "Guangzhou, China",
        "Chongqing, China",
        "Haerbin, China;Harbin, China",
        "Chengdu, China"
    ],
    "South America": [
        "Salvador, Brazil",
        "Belo Horizonte, Brazil",
        "Fortaleza, Brazil",
        "Brasilia, Brazil",
        "Recife, Brazil;Recife City, Brazil",
        "Curitiba, Brazil;Curita, Brazil;Curitiba City, Brazil",
        "Manaus, Brazil;Manaós, Brazil;Manaus City, Brazil",
        "Porto Alegre, Brazil;Porto, Brazil",
        "Concepción, Chile;Concepcion, Chile;Conce, Chile",
        "La Serena, Chile;Serena, Chile;La Serena City, Chile",
        "Temuco, Chile;Temuco City, Chile",
        "Puerto Montt, Chile;Puerto, Chile",
        "Iquique, Chile;Iquique City, Chile",
        "Arica, Chile;Arica City, Chile",
        "Santiago, Chile",
        "Rancagua, Chile;Rancagua City, Chile",
        "Rosario, Argentina;Rosario City, Argentina",
        "Mendoza, Argentina;Mendoza City, Argentina",
        "La Plata, Argentina;La Plata City, Argentina",
        "San Miguel de Tucumán, Argentina;Tucumán, Argentina",
        "Mar del Plata, Argentina;Mardel, Argentina;Mar del Plata City, Argentina",
        "Salta, Argentina;Salta City, Argentina",
        "San Juan, Argentina;San Juan City, Argentina",
        "Neuquén, Argentina;Neuquen, Argentina;Neuquén Capital, Argentina",
        "Cuenca, Ecuador;Santa Ana de los Ríos de Cuenca, Ecuador;Cuenca City, Ecuador",
        "Ambato, Ecuador;Ambato City, Ecuador",
        "Machala, Ecuador;Machala City, Ecuador",
        "Loja, Ecuador;Loja City, Ecuador",
        "Esmeraldas, Ecuador;Esmeraldas City, Ecuador",
        "Riobamba, Ecuador;Riobamba City, Ecuador",
        "Manta, Ecuador;Manta City, Ecuador;Puerto de Manta, Ecuador",
        "Portoviejo, Ecuador;Portoviejo City, Ecuador",
        "Valencia, Venezuela;Valencia del Rey, Venezuela;Valencia City, Venezuela",
        "Barquisimeto, Venezuela;Barquisimeto City, Venezuela",
        "Mérida, Venezuela;Merida, Venezuela",
        "Puerto La Cruz, Venezuela;Puerto, Venezuela;La Cruz, Venezuela",
        "Ciudad Guayana, Venezuela;Cuidad Guayana City, Venezuela",
        "San Cristóbal, Venezuela;San Cristobal, Venezuela",
        "Cumaná, Venezuela;Cumana, Venezuela",
        "Caracas, Venezuela"
    ],
    "Europe": [
        "Milan, Italy;Milano, Italy",
        "Florence, Italy;Firenze, Italy",
        "Venice, Italy;Venezia, Italy",
        "Turin, Italy;Torina, Italy",
        "Bologna, Italy;Bologna City, Italy",
        "Genoa, Italy;Genova, Italy",
        "Palermo, Italy;Panormus, Italy;Palermo City, Italy",
        "Verona, Italy;Verona City, Italy",
        "Dnipro, Ukraine;Dnipropetrovsk, Ukraine;Dnepropetrovsk, Ukraine",
        "Kiev, Ukraine;Kyiv, Ukraine",
        "Lviv, Ukraine;Lvov, Ukraine",
        "Kharkiv, Ukraine;Kharkov, Ukraine",
        "Zaporizhzhia, Ukraine;Zaporozhye, Ukraine",
        "Vinnytsia, Ukraine;Vinnitsa, Ukraine;Vinnytsia City, Ukraine",
        "Mykolaiv, Ukraine;Nikolaev, Ukraine;Mykolaiv City, Ukraine",
        "Ivano-Frankivsk, Ukraine;Stanislav, Ukraine",
        "Birmingham, UK",
        "Glasgow, UK",
        "Edinburgh, UK",
        "Liverpool, UK",
        "Leeds, UK;Leeds City, UK",
        "Bristol, UK;Bristol City, UK",
        "Sheffield, UK;Sheffield City, UK",
        "Cardiff, UK",
        "Timișoara, Romania;Timisoara, Romania",
        "Bucuresti, Romania;Bucharest, Romania",
        "Iași, Romania;Iasi, Romania;Jassy, Romania",
        "Constanța, Romania;Constanta, Romania",
        "Brașov, Romania;Brasov, Romania",
        "Sibiu, Romania;Sibiu City, Romania",
        "Oradea, Romania;Oradea City, Romania",
        "Ploiești, Romania;Ploiesti, Romania;Ploești, Romania;Ploiești City, Romania",
        "Szeged, Hungary",
        "Miskolc, Hungary;Miskolcz, Hungary;Miskolc City, Hungary",
        "Pécs, Hungary;Pecs, Hungary",
        "Nyíregyháza, Hungary;Nyiregyhaza, Hungary;Nyíregyháza City, Hungary",
        "Eger, Hungary;Eger City, Hungary",
        "Budapest, Hungary",
        "Székesfehérvár, Hungary;Szekesfehervar, Hungary;Szfvár, Hungary",
        "Kecskemét, Hungary;Kecskemet, Hungary;Kecskemét City, Hungary"
    ]
}

VALIDATION_CITY_DICT = {
    "Asia": [
        "Seoul, South Korea",
        "Daegu, South Korea;Taegu, South Korea;Taega, South Korea",
        "Delhi, India;New Delhi, India",
        "Bangalore, India;Bengaluru, India",
        "Jakarta, Indonesia;Djakarta, Indonesia",
        "Bagor, Indonesia",
        "Manila, Philippines",
        "Quezon City, Philippines;Quezon, Philippines",
        "Shanghai, China",
        "Beijing, China"
    ],
    "South America": [
        "Sao Paulo, Brazil;São Paulo, Brazil",
        "Rio de Janeiro, Brazil",
        "Valparaíso, Chile;Valparaiso, Chile;Valpo, Chile",
        "Antofagasta, Chile;Antofa, Chile;Antofagasta City, Chile",
        "Buenos Aires, Argentina",
        "Córdoba, Argentina;Cordoba, Argentina",
        "Guayaquil, Ecuador",
        "Quito, Ecuador;San Francisco de Quito, Ecuador",
        "Maracaibo, Venezuela;Maracaibo City, Venezuela",
        "Barcelona, Venezuela;Barcelona City, Venezuela;Barcelona de Venezuela, Venezuela"
    ],
    "Europe": [
        "Rome, Italy",
        "Naples, Italy;Napoli, Italy",
        "Odesa, Ukraine;Odessa, Ukraine",
        "Chernihiv, Ukraine;Chernigov, Ukraine;Chernihiv City, Ukraine",
        "London, UK",
        "Manchester, UK",
        "Cluj-Napoca, Romania;Cluj, Romania;Cluj Napoca, Romania",
        "Galați, Romania;Galati, Romania;Galați City, Romania",
        "Debrecen, Hungary;Debretin, Hungary",
        "Gyor, Hungary;Gyõr, Hungary"
    ]
}

TEST_CITY_DICT = {
    "North America": [
        "Toronto, Canada",
        "Vancouver, Canada",
        "Montreal, Canada;Ville-Marie, Canada",
        "Calgary, Canada;Calgary City, Canada",
        "Ottawa, Canada;Ottawa-Gatineau, Canada",
        "Edmonton, Canada;Edmonton City, Canada",
        "Quebec City, Canada;Québec, Canada;Ville de Québec, Canada",
        "Winnipeg, Canada",
        "Hamilton, Canada;Hamilton City, Canada",
        "Halifax, Canada;Halifax Regional Municipality, Canada",
        "Havanna, Cuba;Havana, Cuba",
        "Santiago de Cuba, Cuba;Santiago, Cuba",
        "Camagüey, Cuba;Puerto Príncipe, Cuba;Camaguey, Cuba",
        "Holguín, Cuba;Holguin, Cuba",
        "Santa Clara, Cuba;Santa Clara City, Cuba",
        "Guantánamo, Cuba;Guantanamo, Cuba;Guantánamo City, Cuba",
        "Cienfuegos, Cuba;Cienfuegos City, Cuba",
        "Bayamo, Cuba;Bayamo City, Cuba",
        "Mantanzas, Cuba;Mantanzas City, Cuba",
        "Pinar del Río, Cuba;Pinar del Rio, Cuba;Pinar City, Cuba"
    ],
    "Africa": [
        "Addis Ababa, Ethiopia",
        "Dire Dawa, Ethiopia;Dire Dhawa, Ethiopia;Diré-Daoua, Ethiopia;Dir Dawa, Ethiopia",
        "Gondar, Ethiopia;Gonder, Ethiopia;Gondär, Ethiopia;Gondaré, Ethiopia",
        "Mekelle, Ethiopia;Mekele, Ethiopia;Mäqälle, Ethiopia",
        "Bahir Dar, Ethiopia;Bahar Dar, Ethiopia;Bahirdar, Ethiopia",
        "Hawassa, Ethiopia;Awassa, Ethiopia;Hawasa, Ethiopia",
        "Jimma, Ethiopia;Jima, Ethiopia;Jimma City, Ethiopia",
        "Harar, Ethiopia;Harer, Ethiopia;Adare, Ethiopia",
        "Dessie, Ethiopia;Dessye, Ethiopia",
        "Adama, Ethiopia;Nazreth, Ethiopia;Nazret, Ethiopia"
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
