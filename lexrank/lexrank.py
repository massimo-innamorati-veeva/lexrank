import math
from collections import Counter, defaultdict

import numpy as np

from lexrank.algorithms.power_method import (
    create_markov_matrix, create_markov_matrix_discrete,
    stationary_distribution, query_biased_stationary_distribution
)
from lexrank.utils.text import tokenize


def degree_centrality_scores(
    similarity_matrix,
    threshold=None,
    increase_power=True,
):
    if not (
        threshold is None
        or isinstance(threshold, float)
        and 0 <= threshold < 1
    ):
        raise ValueError(
            '\'threshold\' should be a floating-point number '
            'from the interval [0, 1) or None',
        )

    if threshold is None:
        markov_matrix = create_markov_matrix(similarity_matrix)

    else:
        markov_matrix = create_markov_matrix_discrete(
            similarity_matrix,
            threshold,
        )

    scores = stationary_distribution(
        markov_matrix,
        increase_power=increase_power,
        normalized=False,
    )

    return scores


def query_biased_degree_centrality_scores(
    similarity_matrix,
    query_relevance_vector,
):
    markov_matrix = create_markov_matrix(similarity_matrix)

    normalized_query_relevance_vector = query_relevance_vector / query_relevance_vector.sum()

    print("normalized query relevance vector:")
    print(normalized_query_relevance_vector)

    print("markov_matrix")
    print(markov_matrix)

    scores = query_biased_stationary_distribution(
        markov_matrix,
        normalized_query_relevance_vector
    )
    return scores, markov_matrix


class LexRank:
    def __init__(
        self,
        documents,
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True,
    ):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words

        self.idf_score = self._calculate_idf(documents)

    def get_summary(
        self,
        sentences,
        summary_size=1,
        threshold=.03,
        fast_power_method=True,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def get_query_focused_summary(
        self,
        sentences,
        query,
        summary_size=1,
        omega=6,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        if query == "" or not query:
            return self.get_summary(sentences, summary_size, threshold=None)

        query_biased_info_scores, markov_matrix = self.rank_sentences_with_query(
            sentences,
            query
        )
        print("query_biased_info_scores")
        print(query_biased_info_scores)
        print("markov_matrix")
        print(markov_matrix)
        exit()

        summary = []
        length = len(sentences)

        if summary_size > length:
            summary_size = length

        affinity_rank_score = np.copy(query_biased_info_scores)

        while True:
            sorted_ix = np.argsort(affinity_rank_score)[::-1]

            top_rank_sentence_idx = sorted_ix[0]

            summary.append(sentences[top_rank_sentence_idx])

            affinity_rank_score[top_rank_sentence_idx] = -10000.0

            for j in range(length):
                affinity_rank_score[j] = affinity_rank_score[j] - omega * markov_matrix[j, top_rank_sentence_idx] * query_biased_info_scores[top_rank_sentence_idx]

            if len(summary) >= summary_size:
                break

        return summary

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        tf_scores = [
            Counter(self.tokenize_sentence(sentence)) for sentence in sentences
        ]

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        scores = degree_centrality_scores(
            similarity_matrix,
            threshold=threshold,
            increase_power=fast_power_method,
        )

        return scores

    def rank_sentences_with_query(
        self,
        sentences,
        query,
    ):
        tf_scores = [
            Counter(self.tokenize_sentence(sentence)) for sentence in sentences
        ]

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        query_tf_score = Counter(self.tokenize_sentence(query))

        print("tokenized query")
        print(self.tokenize_sentence(query))
        print("query_tf_score")
        print(query_tf_score)

        query_relevance_vector = self._calculate_similarity_vector(tf_scores, query_tf_score)

        print("query_relevance_vector")
        print(query_relevance_vector)

        scores, markov_matrix = query_biased_degree_centrality_scores(
            similarity_matrix,
            query_relevance_vector
        )

        return scores, markov_matrix

    def sentences_similarity(self, sentence_1, sentence_2):
        tf_1 = Counter(self.tokenize_sentence(sentence_1))
        tf_2 = Counter(self.tokenize_sentence(sentence_2))

        similarity = self._idf_modified_cosine([tf_1, tf_2], 0, 1)

        return similarity

    def tokenize_sentence(self, sentence):
        tokens = tokenize(
            sentence,
            self.stopwords,
            keep_numbers=self.keep_numbers,
            keep_emails=self.keep_emails,
            keep_urls=self.keep_urls,
        )

        return tokens

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = 1

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def _calculate_similarity_matrix(self, tf_scores):
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _calculate_similarity_vector(self, tf_scores, query_tf_score):
        length = len(tf_scores)

        similarity_vector = np.zeros([length])

        for i in range(length):
            similarity = self._idf_modified_cosine_one_pair(tf_scores[i], query_tf_score)

            similarity_vector[i] = similarity

        return similarity_vector

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

    def _idf_modified_cosine_one_pair(self, tf_i, tf_j):
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity
