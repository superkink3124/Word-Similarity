from scipy import stats, spatial
import numpy as np


class WordSimilarity:
    @staticmethod
    def cosine(e1, e2):
        return 1 - spatial.distance.cosine(e1, e2)

    @staticmethod
    def pearson(e1, e2):
        return stats.pearsonr(e1, e2)[0]

    @staticmethod
    def spearman_rank(e1, e2):
        return stats.spearmanr(e1, e2)[0]

    # @staticmethod
    # def dot_product(e1, e2):
    #     return np.dot(e1, e2)

    @staticmethod
    def euclidean(e1, e2):
        return 1 - spatial.distance.euclidean(e1, e2)
