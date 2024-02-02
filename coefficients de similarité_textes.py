pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import FreqDist
import string

nltk.download('stopwords')

"latin stopwords = ab, ac, ad, adhic, aliqui, aliquis, an, ante, apud, at, atque, aut, autem, cum, cur, de, deinde, dum, ego, enim, ergo, es, est, et, etiam, etsi, ex, fio, haud, hic, iam, idem, igitur, ille, in, infra, inter, interim, ipse, is, ita, magis, modo, mox, nam, ne, nec, necque, neque, nisi, non, nos, o, ob, per, possum, post, pro, quae, quam, quare, qui, quia, quicumque, quidem, quilibet, quis, quisnam, quisquam, quisque, quisquis, quo, quoniam, sed, si, sic, sive, sub, sui, sum, super, suus, tam, tamen, trans, tu, tum, ubi, vel, uel, uero"

nltk.download('punkt')

def load_text_database(directory_path):
    # Implement loading texts from your database
    pass

text_database = load_text_database('/emplacement de la base de donnees textuelles')

#pre-traitement du texte

def preprocess_text(text):
    stop_words = set(stopwords.words('latin'))
    ps = PorterStemmer()

    # Tokenization
    words = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words and word not in string.punctuation]

    return words


def preprocess_database(text_database):
    preprocessed_database = {}

    for text_id, text in text_database.items():
        preprocessed_database[text_id] = preprocess_text(text)

    return preprocessed_database

#Parametrage de la fonction de calcul de similarite

def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1)
    set2 = set(text2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0.0


#Calcul de la similarite

def calculate_similarity(input_text, text_database):
    preprocessed_input_text = preprocess_text(input_text)
    preprocessed_database = preprocess_database(text_database)

    similarities = {}

    for text_id, text in preprocessed_database.items():
        similarity = calculate_jaccard_similarity(preprocessed_input_text, text)
        similarities[text_id] = similarity

    return similarities







