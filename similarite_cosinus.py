import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

nltk.download('stopwords')

"latin stopwords = ab, ac, ad, adhic, aliqui, aliquis, an, ante, apud, at, atque, aut, autem, cum, cur, de, deinde, dum, ego, enim, ergo, es, est, et, etiam, etsi, ex, fio, haud, hic, iam, idem, igitur, ille, in, infra, inter, interim, ipse, is, ita, magis, modo, mox, nam, ne, nec, necque, neque, nisi, non, nos, o, ob, per, possum, post, pro, quae, quam, quare, qui, quia, quicumque, quidem, quilibet, quis, quisnam, quisquam, quisque, quisquis, quo, quoniam, sed, si, sic, sive, sub, sui, sum, super, suus, tam, tamen, trans, tu, tum, ubi, vel, uel, uero"

nltk.download('punkt')

def load_text_database(directory_path):
    # charge les textes depuis la base de donnees
    pass

def preprocess_text(text):
    stop_words = set(stopwords.words('latin'))
    ps = PorterStemmer()

    # tokenise
    words = word_tokenize(text.lower())

    # supprime ponctuation et stop words
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words and word not in string.punctuation]

    return ' '.join(words)

def preprocess_database(text_database):
    preprocessed_texts = []

    for text_id, text in text_database.items():
        preprocessed_texts.append(preprocess_text(text))

    return preprocessed_texts

def calculate_jaccard_similarity(text1, text2):
    set1 = set(text1)
    set2 = set(text2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0.0

def calculate_cosine_similarity(input_text, text_database):
    preprocessed_input_text = preprocess_text(input_text)
    preprocessed_database = preprocess_database(text_database)

    # combine texte d'entree et textes de la base
    all_texts = [preprocessed_input_text] + preprocessed_database

    # genere la matrice
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(all_texts)

    # calcul la similarite cosinus entre le texte d'entree et les textes de la base
    similarities = cosine_similarity(count_matrix)[0][1:]

    return {text_id: similarity for text_id, similarity in zip(text_database.keys(), similarities)}

def calculate_similarity(input_text, text_database):
    jaccard_similarities = {}
    cosine_similarities = {}

    preprocessed_input_text = preprocess_text(input_text)
    preprocessed_database = preprocess_database(text_database)

    for text_id, text in enumerate(preprocessed_database):
        jaccard_similarity = calculate_jaccard_similarity(preprocessed_input_text.split(), text.split())
        jaccard_similarities[text_id] = jaccard_similarity

    cosine_similarities = calculate_cosine_similarity(input_text, text_database)

    return jaccard_similarities, cosine_similarities

text_database = load_text_database('/path/to/text_database')

input_text = "texte d'entree."

jaccard_similarities, cosine_similarities = calculate_similarity(input_text, text_database)

print("Jaccard Similarities:")
print(jaccard_similarities)
print("\nCosine Similarities:")
print(cosine_similarities)