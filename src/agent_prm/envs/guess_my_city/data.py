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
        "Pune, India;Poona, India",
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
        "Kyoto, Japan;Kyōto, Japan;Miyako, Japan",
        "Hiroshima, Japan;Hiroshima-shi, Japan",
        "Sapporo, Japan;Sapporo-shi, Japan",
        "Fukuoka, Japan;Fukuoka-shi, Japan",
        "Kobe, Japan;Kōbe, Japan;Kobe-shi, Japan",
        "Nagasaki, Japan;Nagasaki-shi, Japan;Nangasaki, Japan"
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
        "Paris, France",
        "Marseille, France;Marseilles, France;Massalia, France",
        "Lyon, France;Lyons, France;Lugdunum, France",
        "Toulouse, France",
        "Nice, France",
        "Nantes, France;Nantes City, France",
        "Strasbourg, France",
        "Lille, France",
        "Bordeaux, France",
        "Montpellier, France;Montpellier City, France",
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
        "Gyor, Hungary",
        "Nyíregyháza, Hungary;Nyiregyhaza, Hungary;Nyíregyháza City, Hungary"
    ],
    "North America": [
        "Mexico City, Mexico",
        "Guadalajara, Mexico;Guad, Mexico",
        "Monterrey, Mexico;Monterrei, Mexico",
        "Puebla, Mexico;Puebla de Zaragoza, Mexico;Puebla City, Mexico",
        "Tijuana, Mexico;Tijuana City, Mexico;Tía Juana, Mexico",
        "Mérida, Mexico;Merida, Mexico;Mérida de Yicatán, Mexico",
        "León, Mexico;Leon, Mexico;León de los Aldama, Mexico;León City, Mexico",
        "San Luis Potosí, Mexico;San Luis, Mexico;San Luis Potosi, Mexico",
        "Cancún, Mexico;Cancun, Mexico;Cancún City, Mexico",
        "Toluca, Mexico;Toluca de Lerdo, Mexico;Toluca City, Mexico",
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "San Francisco, USA",
        "Miami, USA;Miami City, USA",
        "Seattle, USA;Seattle City, USA",
        "Boston, USA;Boston City, USA",
        "Atlanta, USA",
        "Philadelphia, USA",
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
        "Cairo, Egypt",
        "Alexandria, Egypt",
        "Giza, Egypt",
        "Luxor, Egypt",
        "Aswan, Egypt;Syene, Egypt",
        "Port Said, Egypt;Port Saeed, Egypt;Port Said City, Egypt",
        "Suez, Egypt;Suez City, Egypt;Port of Suez, Egypt",
        "Mansoura, Egypt;Mansura, Egypt;El Mansoura, Egypt",
        "Tanta, Egypt;Tanta City, Egypt",
        "Faiyum, Egypt;Fayoum, Egypt;Faiyum City, Egypt",
        "Casablanca, Morocco",
        "Marrakesh, Morocco;Marrakech, Morocco",
        "Fes, Morocco;Fez, Morocco",
        "Rabat, Morocco;Rabat City, Morocco",
        "Tangier, Morocco",
        "Agadir, Morocco;Agadir City, Morocco",
        "Oujda, Morocco;Wajda, Morocco;Oujda City, Morocco",
        "Meknes, Morocco;Meknes City, Morocco",
        "Tétouan, Morocco;Tetouan, Morocco",
        "El Jadida, Morocco;El Jadida City, Morocco",
        "Kinshaha, Congo;Kinshasa, Congo",
        "Lubumbashi, Congo;Élisabethville, Congo;Lumbumbashi, Congo",
        "Mbuji-Mayi, Congo;Mbujimayi, Congo;Mbuji Mayi, Congo;Bakwanga, Congo",
        "Kisangani, Congo;Kisangani City, Congo",
        "Goma, Congo;Goma City, Congo",
        "Brazzaville, Congo;Ville de Brazzaville, Congo",
        "Pointe-Noire, Congo;Point-Noire, Congo",
        "Matadi, Congo;Matadi City, Congo;Port de Matadi, Congo",
        "Bukavu, Congo;Bukavu City, Congo",
        "Dolisie, Congo;Dolisie City, Congo",
        "Addis Ababa, Ethiopia",
        "Dire Dawa, Ethiopia;Dire Dhawa, Ethiopia;Diré-Daoua, Ethiopia;Dir Dawa, Ethiopia",
        "Gondar, Ethiopia;Gonder, Ethiopia;Gondär, Ethiopia;Gondaré, Ethiopia",
        "Mekelle, Ethiopia;Mekele, Ethiopia;Mäqälle, Ethiopia",
        "Bahir Dar, Ethiopia;Bahar Dar, Ethiopia;Bahirdar, Ethiopia",
        "Hawassa, Ethiopia;Awassa, Ethiopia;Hawasa, Ethiopia",
        "Jimma, Ethiopia;Jima, Ethiopia;Jimma City, Ethiopia",
        "Harar, Ethiopia;Harer, Ethiopia;Adare, Ethiopia",
        "Dessie, Ethiopia;Dessye, Ethiopia",
        "Adama, Ethiopia;Nazreth, Ethiopia;Nazret, Ethiopia",
        "Abidjan, Cote d'Ivorie;Abidjan, Côte d'Ivoire",
        "Yamoussoukro, Cote d'Ivorie;Yamoussukro, Cote d'Ivorie;Yamassoukro, Cote d'Ivorie;Yamousouko, Cote d'Ivorie",
        "Bouaké, Cote d'Ivorie;Bouake, Cote d'Ivorie;Bwake, Cote d'Ivorie;Bouaké City, Cote d'Ivorie",
        "Daloa, Cote d'Ivorie;Daloa City, Cote d'Ivorie",
        "San Pedro, Cote d'Ivorie;San-Pédro, Cote d'Ivorie",
        "Korhogo, Cote d'Ivorie;Korhogo City, Cote d'Ivorie",
        "Man, Cote d'Ivorie;Man City, Cote d'Ivorie",
        "Gagnoa, Cote d'Ivorie;Gagnoa City, Cote d'Ivorie",
        "Bondoukou, Cote d'Ivorie;Bondoukou City, Cote d'Ivorie",
        "Odienné, Cote d'Ivorie;Odienne, Cote d'Ivorie;Odienné City, Cote d'Ivorie"
    ],
    "Australia": [
        "Sydney, Australia",
        "Melbourne, Australia;Melburn, Australia;Melbourne City, Australia",
        "Brisbane, Australia;Brisbane City, Australia",
        "Perth, Australia;Perth City, Australia",
        "Adelaide, Australia",
        "Canberra, Australia",
        "Hobart, Australia",
        "Darwin, Australia;Darwin City, Australia",
        "Gold Coast, Australia",
        "Newcastle, Australia;Newcastle City, Australia"
    ]
}

TRAIN_CITY_DICT = {
    "Asia": [
        "Incheon, South Korea;Inchon, South Korea",
        "Daejeon, South Korea;Taejon, South Korea;Daejeon-si, South Korea",
        "Gwangju, South Korea;Kwangju, South Korea;Gwangju Metropolitan City, South Korea",
        "Suwon, South Korea;Suwon-si, South Korea",
        "Ulsan, South Korea;Ulsan Metropolitan City, South Korea",
        "Jeonju, South Korea;Chǒnju, South Korea;Jeonju-si, South Korea",
        "Gyeongju, South Korea;Kyǒngju, South Korea;Geyongju-si, South Korea;Seorabeol, South Korea",
        "Madras, India;Chennai, India",
        "Bangalore, India;Bengaluru, India",
        "Hyderabad, India",
        "Ahmedabad, India",
        "Kanpur, India",
        "Bengaluru, India;Bangalore, India;Bengalooru, India",
        "Chennai, India;Madras, India",
        "Pune, India;Poona, India",
        "Malang, Indonesia",
        "Surabaya, Indonesia",
        "Semarang, Indonesia",
        "Sukabumi, Indonesia",
        "Cirebon, Indonesia",
        "Medan, Indonesia",
        "Yogyakarta, Indonesia;Jogjakarta, Indonesia;Jogja, Indonesia;Kota Yogyakarta, Indonesia",
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
        "Nagoya, Japan",
        "Kyoto, Japan;Kyōto, Japan;Miyako, Japan",
        "Hiroshima, Japan;Hiroshima-shi, Japan",
        "Sapporo, Japan;Sapporo-shi, Japan",
        "Fukuoka, Japan;Fukuoka-shi, Japan",
        "Kobe, Japan;Kōbe, Japan;Kobe-shi, Japan",
        "Nagasaki, Japan;Nagasaki-shi, Japan;Nangasaki, Japan"
    ],
    "South America": [
        "Belo Horizonte, Brazil",
        "Fortaleza, Brazil",
        "Brasilia, Brazil",
        "Recife, Brazil;Recife City, Brazil",
        "Curitiba, Brazil;Curita, Brazil;Curitiba City, Brazil",
        "Manaus, Brazil;Manaós, Brazil;Manaus City, Brazil",
        "Porto Alegre, Brazil;Porto, Brazil",
        "La Serena, Chile;Serena, Chile;La Serena City, Chile",
        "Antofagasta, Chile;Antofa, Chile;Antofagasta City, Chile",
        "Temuco, Chile;Temuco City, Chile",
        "Puerto Montt, Chile;Puerto, Chile",
        "Iquique, Chile;Iquique City, Chile",
        "Arica, Chile;Arica City, Chile",
        "Rancagua, Chile;Rancagua City, Chile",
        "Mendoza, Argentina;Mendoza City, Argentina",
        "La Plata, Argentina;La Plata City, Argentina",
        "San Miguel de Tucumán, Argentina;Tucumán, Argentina",
        "Mar del Plata, Argentina;Mardel, Argentina;Mar del Plata City, Argentina",
        "Salta, Argentina;Salta City, Argentina",
        "San Juan, Argentina;San Juan City, Argentina",
        "Neuquén, Argentina;Neuquen, Argentina;Neuquén Capital, Argentina",
        "Ambato, Ecuador;Ambato City, Ecuador",
        "Machala, Ecuador;Machala City, Ecuador",
        "Loja, Ecuador;Loja City, Ecuador",
        "Esmeraldas, Ecuador;Esmeraldas City, Ecuador",
        "Riobamba, Ecuador;Riobamba City, Ecuador",
        "Manta, Ecuador;Manta City, Ecuador;Puerto de Manta, Ecuador",
        "Portoviejo, Ecuador;Portoviejo City, Ecuador",
        "Barquisimeto, Venezuela;Barquisimeto City, Venezuela",
        "Mérida, Venezuela;Merida, Venezuela",
        "Puerto La Cruz, Venezuela;Puerto, Venezuela;La Cruz, Venezuela",
        "Ciudad Guayana, Venezuela;Cuidad Guayana City, Venezuela",
        "San Cristóbal, Venezuela;San Cristobal, Venezuela",
        "Barcelona, Venezuela;Barcelona City, Venezuela;Barcelona de Venezuela, Venezuela",
        "Cumaná, Venezuela;Cumana, Venezuela"
    ],
    "Europe": [
        "Florence, Italy;Firenze, Italy",
        "Venice, Italy;Venezia, Italy",
        "Turin, Italy;Torina, Italy",
        "Bologna, Italy;Bologna City, Italy",
        "Genoa, Italy;Genova, Italy",
        "Palermo, Italy;Panormus, Italy;Palermo City, Italy",
        "Verona, Italy;Verona City, Italy",
        "Lviv, Ukraine;Lvov, Ukraine",
        "Kharkiv, Ukraine;Kharkov, Ukraine",
        "Zaporizhzhia, Ukraine;Zaporozhye, Ukraine",
        "Vinnytsia, Ukraine;Vinnitsa, Ukraine;Vinnytsia City, Ukraine",
        "Mykolaiv, Ukraine;Nikolaev, Ukraine;Mykolaiv City, Ukraine",
        "Chernihiv, Ukraine;Chernigov, Ukraine;Chernihiv City, Ukraine",
        "Ivano-Frankivsk, Ukraine;Stanislav, Ukraine",
        "Toulouse, France",
        "Nice, France",
        "Nantes, France;Nantes City, France",
        "Strasbourg, France",
        "Lille, France",
        "Bordeaux, France",
        "Montpellier, France;Montpellier City, France",
        "Iași, Romania;Iasi, Romania;Jassy, Romania",
        "Constanța, Romania;Constanta, Romania",
        "Brașov, Romania;Brasov, Romania",
        "Sibiu, Romania;Sibiu City, Romania",
        "Oradea, Romania;Oradea City, Romania",
        "Galați, Romania;Galati, Romania;Galați City, Romania",
        "Ploiești, Romania;Ploiesti, Romania;Ploești, Romania;Ploiești City, Romania",
        "Miskolc, Hungary;Miskolcz, Hungary;Miskolc City, Hungary",
        "Pécs, Hungary;Pecs, Hungary",
        "Gyor, Hungary",
        "Nyíregyháza, Hungary;Nyiregyhaza, Hungary;Nyíregyháza City, Hungary"
    ]
}

VALIDATION_CITY_DICT = {
    "Asia": [
        "Seoul, South Korea",
        "Busan, South Korea;Pusan, South Korea",
        "Daegu, South Korea;Taegu, South Korea;Taega, South Korea",
        "Bombay, India;Mumbai, India",
        "Delhi, India;New Delhi, India",
        "Calcutta, India;Kolkata, India",
        "Jakarta, Indonesia;Djakarta, Indonesia",
        "Bandung, Indonesia",
        "Bagor, Indonesia",
        "Shanghai, China",
        "Beijing, China",
        "Hong Kong, China",
        "Tokyo, Japan",
        "Yokohama, Japan",
        "Osaka, Japan"
    ],
    "South America": [
        "Sao Paulo, Brazil;São Paulo, Brazil",
        "Rio de Janeiro, Brazil",
        "Salvador, Brazil",
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
    ]
}

TEST_CITY_DICT = {
    "North America": [
        "Mexico City, Mexico",
        "Guadalajara, Mexico;Guad, Mexico",
        "Monterrey, Mexico;Monterrei, Mexico",
        "Puebla, Mexico;Puebla de Zaragoza, Mexico;Puebla City, Mexico",
        "Tijuana, Mexico;Tijuana City, Mexico;Tía Juana, Mexico",
        "Mérida, Mexico;Merida, Mexico;Mérida de Yicatán, Mexico",
        "León, Mexico;Leon, Mexico;León de los Aldama, Mexico;León City, Mexico",
        "San Luis Potosí, Mexico;San Luis, Mexico;San Luis Potosi, Mexico",
        "Cancún, Mexico;Cancun, Mexico;Cancún City, Mexico",
        "Toluca, Mexico;Toluca de Lerdo, Mexico;Toluca City, Mexico",
        "New York, USA",
        "Los Angeles, USA",
        "Chicago, USA",
        "Houston, USA",
        "San Francisco, USA",
        "Miami, USA;Miami City, USA",
        "Seattle, USA;Seattle City, USA",
        "Boston, USA;Boston City, USA",
        "Atlanta, USA",
        "Philadelphia, USA",
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
        "Cairo, Egypt",
        "Alexandria, Egypt",
        "Giza, Egypt",
        "Luxor, Egypt",
        "Aswan, Egypt;Syene, Egypt",
        "Port Said, Egypt;Port Saeed, Egypt;Port Said City, Egypt",
        "Suez, Egypt;Suez City, Egypt;Port of Suez, Egypt",
        "Mansoura, Egypt;Mansura, Egypt;El Mansoura, Egypt",
        "Tanta, Egypt;Tanta City, Egypt",
        "Faiyum, Egypt;Fayoum, Egypt;Faiyum City, Egypt",
        "Casablanca, Morocco",
        "Marrakesh, Morocco;Marrakech, Morocco",
        "Fes, Morocco;Fez, Morocco",
        "Rabat, Morocco;Rabat City, Morocco",
        "Tangier, Morocco",
        "Agadir, Morocco;Agadir City, Morocco",
        "Oujda, Morocco;Wajda, Morocco;Oujda City, Morocco",
        "Meknes, Morocco;Meknes City, Morocco",
        "Tétouan, Morocco;Tetouan, Morocco",
        "El Jadida, Morocco;El Jadida City, Morocco",
        "Kinshaha, Congo;Kinshasa, Congo",
        "Lubumbashi, Congo;Élisabethville, Congo;Lumbumbashi, Congo",
        "Mbuji-Mayi, Congo;Mbujimayi, Congo;Mbuji Mayi, Congo;Bakwanga, Congo",
        "Kisangani, Congo;Kisangani City, Congo",
        "Goma, Congo;Goma City, Congo",
        "Brazzaville, Congo;Ville de Brazzaville, Congo",
        "Pointe-Noire, Congo;Point-Noire, Congo",
        "Matadi, Congo;Matadi City, Congo;Port de Matadi, Congo",
        "Bukavu, Congo;Bukavu City, Congo",
        "Dolisie, Congo;Dolisie City, Congo",
        "Addis Ababa, Ethiopia",
        "Dire Dawa, Ethiopia;Dire Dhawa, Ethiopia;Diré-Daoua, Ethiopia;Dir Dawa, Ethiopia",
        "Gondar, Ethiopia;Gonder, Ethiopia;Gondär, Ethiopia;Gondaré, Ethiopia",
        "Mekelle, Ethiopia;Mekele, Ethiopia;Mäqälle, Ethiopia",
        "Bahir Dar, Ethiopia;Bahar Dar, Ethiopia;Bahirdar, Ethiopia",
        "Hawassa, Ethiopia;Awassa, Ethiopia;Hawasa, Ethiopia",
        "Jimma, Ethiopia;Jima, Ethiopia;Jimma City, Ethiopia",
        "Harar, Ethiopia;Harer, Ethiopia;Adare, Ethiopia",
        "Dessie, Ethiopia;Dessye, Ethiopia",
        "Adama, Ethiopia;Nazreth, Ethiopia;Nazret, Ethiopia",
        "Abidjan, Cote d'Ivorie;Abidjan, Côte d'Ivoire",
        "Yamoussoukro, Cote d'Ivorie;Yamoussukro, Cote d'Ivorie;Yamassoukro, Cote d'Ivorie;Yamousouko, Cote d'Ivorie",
        "Bouaké, Cote d'Ivorie;Bouake, Cote d'Ivorie;Bwake, Cote d'Ivorie;Bouaké City, Cote d'Ivorie",
        "Daloa, Cote d'Ivorie;Daloa City, Cote d'Ivorie",
        "San Pedro, Cote d'Ivorie;San-Pédro, Cote d'Ivorie",
        "Korhogo, Cote d'Ivorie;Korhogo City, Cote d'Ivorie",
        "Man, Cote d'Ivorie;Man City, Cote d'Ivorie",
        "Gagnoa, Cote d'Ivorie;Gagnoa City, Cote d'Ivorie",
        "Bondoukou, Cote d'Ivorie;Bondoukou City, Cote d'Ivorie",
        "Odienné, Cote d'Ivorie;Odienne, Cote d'Ivorie;Odienné City, Cote d'Ivorie"
    ],
    "Australia": [
        "Sydney, Australia",
        "Melbourne, Australia;Melburn, Australia;Melbourne City, Australia",
        "Brisbane, Australia;Brisbane City, Australia",
        "Perth, Australia;Perth City, Australia",
        "Adelaide, Australia",
        "Canberra, Australia",
        "Hobart, Australia",
        "Darwin, Australia;Darwin City, Australia",
        "Gold Coast, Australia",
        "Newcastle, Australia;Newcastle City, Australia"
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
