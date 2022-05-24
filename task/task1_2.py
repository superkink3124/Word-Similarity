from gensim.models import KeyedVectors
import pandas as pd
from task.word_similarity import WordSimilarity
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_similarity():
    global word2vec, vocab
    visim_400 = pd.read_csv("datasets/ViSim-400/Visim-400.txt", delimiter="\t")
    cosine_sim = []
    pearson_sim = []
    spearman_sim = []
    for _, row in visim_400.iterrows():
        word1 = row["Word1"]
        word2 = row["Word2"]
        if word1 in vocab and word2 in vocab:
            cosine_sim.append(round(WordSimilarity.cosine(word2vec[word1], word2vec[word2]), 3))
            pearson_sim.append(round(WordSimilarity.pearson(word2vec[word1], word2vec[word2]), 3))
            spearman_sim.append(round(WordSimilarity.spearman_rank(word2vec[word1], word2vec[word2]), 3))
        else:
            cosine_sim.append("None")
            pearson_sim.append("None")
            spearman_sim.append("None")
    visim_400["Cosine"] = cosine_sim
    visim_400["Pearson"] = pearson_sim
    visim_400["Spearman "] = spearman_sim
    visim_400.to_csv("task/visim-400.csv", index=False)


def k_nearest_neighbor(query_word, k, sim_function):
    global word2vec, vocab
    candidates = []
    for word in vocab:
        candidates.append((sim_function(word2vec[word], word2vec[query_word]), word))
    candidates.sort(reverse=True)
    candidates = [candidate[1] for candidate in candidates[:k]]
    return candidates


def show_mean_plot():
    global word2vec, vocab
    mean_embedding = []
    for word in vocab:
        mean_embedding.append(word2vec[word].sum() / 150)
    sns.displot(mean_embedding)
    plt.show()


if __name__ == '__main__':
    word2vec = KeyedVectors.load_word2vec_format('word2vec/W2V_150.txt')
    vocab = word2vec.index_to_key
    show_mean_plot()
    evaluate_similarity()
    print(k_nearest_neighbor("khử_trùng", 10, WordSimilarity.cosine))
    print(k_nearest_neighbor("khử_trùng", 10, WordSimilarity.pearson))
    print(k_nearest_neighbor("khử_trùng", 10, WordSimilarity.spearman_rank))
