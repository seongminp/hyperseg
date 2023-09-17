import argparse

import kmedoids
import torchhd

# from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# from sklearn.neighbors import NearestCentroid
# from sklearn_extra.cluster import KMedoids

from .embedder import Embedder


class Summarizer:
    def __init__(self, clustering_algorithm="alternating", tokenizer=None, vsa_type=None):
        self.embedder = Embedder(tokenizer=tokenizer, vsa_type=vsa_type)
        self.clustering_algorithm = clustering_algorithm

    def summarize(self, sentences, n):
        if n <= 0:
            return []

        if n > len(sentences):
            return sentences

        embeddings, sentence_index = self.embedder.embed(sentences)
        distances = pairwise_distances(embeddings, metric="cosine")
        if self.clustering_algorithm == "fasterpam": 
            medoids = kmedoids.fasterpam(distances, n, random_state=0, init="build")
        elif self.clustering_algorithm == "fastpam1": 
            medoids = kmedoids.fastpam1(distances, n, random_state=0, init="build")
        elif self.clustering_algorithm == "fastermsc": 
            medoids = kmedoids.fastermsc(distances, n, random_state=0, init="build")
        elif self.clustering_algorithm == "fastmsc": 
            medoids = kmedoids.fastmsc(distances, n, random_state=0, init="build")
        elif self.clustering_algorithm == "alternating": 
            medoids = kmedoids.alternating(distances, n, random_state=0, init="build")
        else:
            raise ValueError("Wrong algorithm:", self.clustering_algotithm)

        centers = medoids.medoids
        centers.sort()
        summaries = [sentences[sentence_index[i]] for i in centers]
        return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()

