import argparse

import numpy as np
import torch
import torchhd

from .stopwords import stopwords


def n_grams(string, n):
    return [string[i : i + n] for i in range(len(string) - n + 1)]


def get_local_minima(array, threshold=None):
    local_minima_indices = []
    local_minima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] > array[i] and array[i] < array[i + 1]:
            if threshold is None or array[i] > threshold:
                local_minima_indices.append(i)
                local_minima_values.append(array[i])
    return local_minima_indices, local_minima_values


class HyperSegSegmenter:
    def __init__(self, dimension=10000, ngrams=None, tokenizer=None):
        self.dimension = dimension
        self.tokenizer = tokenizer
        self.ngrams = ngrams

    def segment(self, utterances, damp=False):
        if not utterances:
            return [], [], [], []

        torch.manual_seed(0)

        word_list, word_to_index = [], {}
        tokenized_utterances = []

        original_index = {}  # We only want to collect vectors from valid utterances.
        for ui, utterance in enumerate(utterances):
            if not utterance.strip():
                print(f"Empty utterance: {utterance}")
                continue

            utterance = utterance.lower()
            words = (
                utterance.split()
                if self.ngrams is None
                else n_grams(utterance, self.ngrams)
            )
            tokenized_utterance = []
            for word in words:
                if word in stopwords and self.ngrams is None:
                    continue

                if word in word_to_index:
                    word_id = word_to_index[word]
                else:
                    word_id = len(word_list)
                    word_to_index[word] = word_id
                    word_list.append(word)
                tokenized_utterance.append(word_id)
            if not tokenized_utterance:
                print(f"No token utterance: {utterance}, ngram={self.ngrams}")
                continue
            original_index[len(tokenized_utterances)] = ui
            tokenized_utterances.append(tokenized_utterance)

        device = "cpu"
        token_embeddings = torchhd.circular(
            len(word_list), self.dimension, device=device
        )

        utterance_vectors = []
        for tokenized_utterance in tokenized_utterances:
            symbols = token_embeddings[tokenized_utterance]
            utterance_vector = torchhd.bundle_sequence(symbols)
            utterance_vectors.append(utterance_vector)

        sims = [
            torchhd.cosine_similarity(prev, curr)
            for prev, curr in zip(utterance_vectors[:-1], utterance_vectors[1:])
        ]
        sims = torch.tensor(sims, device=device)

        threshold = sims.mean() - sims.std()
        minima_indices, minima_values = get_local_minima(sims, threshold)
        if damp:
            choices = (
                int(np.log(len(minima_indices)) ** 2) if len(minima_indices) > 0 else 0
            )
            minima_argmin = torch.argsort(torch.tensor(minima_values))[:choices]
            segment_indices = [minima_indices[i] for ii, i in enumerate(minima_argmin)]
        else:
            segment_indices = minima_indices

        segments = [0] * (len(utterances) - 1)
        for index in segment_indices:
            segments[original_index[index]] = 1

        return segments, utterance_vectors, threshold, sims

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()

