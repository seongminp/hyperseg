import argparse
import nltk

import numpy as np
import torch
import torchhd
from nltk.corpus import stopwords
from torchhd import ensure_vsa_tensor
from transformers import AutoTokenizer

nltk.download('averaged_perceptron_tagger')

class WordTokenizer:
    def tokenize(self, sentence):
        return [w.strip().lower() for w in sentence.split()]


def get_ngrams(string, n):
    return [string[i : i + n] for i in range(len(string) - n + 1)]


class NGramTokenizer:
    def __init__(self, n):
        self.n = n

    def tokenize(self, sentence):
        return get_ngrams(sentence.lower(), self.n)


class BertTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

class POSTokenizer:
    def tokenize(self, sentence):
        tagged = nltk.pos_tag([sentence])
        tokenized = [word for word, _ in tagged]
        return tokenized


class Embedder:
    def __init__(self, tokenizer=None, dimension=10000, vsa_type=None):
        if tokenizer is None:
            tokenizer = WordTokenizer()
        elif tokenizer == 'word':
            tokenizer = WordTokenizer()
        elif tokenizer == 'ngram2':
            tokenizer = NGramTokenizer(2)
        elif tokenizer == 'ngram3':
            tokenizer = NGramTokenizer(3)
        elif tokenizer == 'ngram5':
            tokenizer = NGramTokenizer(5)
        elif tokenizer == 'bert':
            tokenizer = BertTokenizer()
        elif tokenizer == 'pos':
            tokenizer = POSTokenizer()
        else:
            raise ValueError(f"Wrong tokenizer name: {tokenizer}")

        if vsa_type is None:
            self.embedder = torchhd.thermometer
        elif vsa_type == 'thermometer':
            self.embedder = torchhd.thermometer
        elif vsa_type == 'random':
            self.embedder = torchhd.random
        elif vsa_type == 'level':
            self.embedder = torchhd.level
        elif vsa_type == 'circular':
            self.embedder = torchhd.circular
        else:
            raise ValueError(f"Wrong vsa_type: {vsa_type}")


        self.tokenizer = tokenizer
        self.dimension = dimension
        self.stopwords = set(stopwords.words("english"))

    def embed(self, sentences, precomputed_token_embeddings=None):
        token_to_id = {}
        tokenized_sentences = []

        if precomputed_token_embeddings is not None:
            token_embeddings = precomputed_token_embeddings

        else:
            tokens = list(
                {
                    token
                    for sentence in sentences
                    for token in self.tokenizer.tokenize(sentence)
                }
            )
            tokens.append("oov")
            random_embeddings = self.embedder( 
                len(tokens), self.dimension, device="cpu"
            )
            token_embeddings = {
                token: random_embeddings[ti].numpy() for ti, token in enumerate(tokens)
            }

        embeddings = []
        embedding_index, sentence_index = 0, {}
        for i, sentence in enumerate(sentences):
            symbols = np.array(
                [
                    token_embeddings[token]
                    for token in self.tokenizer.tokenize(sentence)
                    if token not in self.stopwords
                ]
            )
            if symbols.shape[0] > 0:
                embedding = torchhd.bundle_sequence(symbols)
                embeddings.append(embedding.numpy))
                sentence_index[embedding_index] = i
                embedding_index += 1

        embeddings = np.vstack(embeddings)

        return embeddings, sentence_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
(
