from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from typing import List

import time


class ItemitemWithKNNRec:
    def __init__(self):
        self.ratings = pd.read_csv("ratings_large.csv")
        self.movies = pd.read_csv("movies_large.csv")
        self.movie_titles = dict(zip(self.movies['movieId'], self.movies['title']))

        self.X, self.user_mapper, self.movie_mapper, self.user_inv_mapper, self.movie_inv_mapper = self.create_X(
            self.ratings)

        self.svd = TruncatedSVD(n_components=20, n_iter=10)
        self.Q = self.svd.fit_transform(self.X.T)

    def create_X(self, df):
        """
        Generates a sparse matrix from ratings dataframe.

        Args:
            df: pandas dataframe containing 3 columns (userId, movieId, rating)

        Returns:
            X: sparse matrix
            user_mapper: dict that maps user id's to user indices
            user_inv_mapper: dict that maps user indices to user id's
            movie_mapper: dict that maps movie id's to movie indices
            movie_inv_mapper: dict that maps movie indices to movie id's
        """
        M = df['userId'].nunique()
        N = df['movieId'].nunique()

        user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
        movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

        user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
        movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

        user_index = [user_mapper[i] for i in df['userId']]
        item_index = [movie_mapper[i] for i in df['movieId']]

        X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

        return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def find_similar_movies(self, movie_id, X, movie_mapper, movie_inv_mapper, k, metric) -> List:
        """
        Finds k-nearest neighbours for a given movie id.

        Args:
            movie_id: id of the movie of interest
            X: user-item utility matrix
            k: number of similar movies to retrieve
            metric: distance metric for kNN calculations

        Output: returns list of k similar movie ID's
        """
        X = X.T
        neighbour_ids = []

        movie_ind = movie_mapper[movie_id]
        movie_vec = X[movie_ind]
        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1, -1)
        # use k+1 since kNN output includes the movieId of interest
        kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric=metric)
        kNN.fit(X)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)
        for i in range(0, k):
            n = neighbour.item(i)
            neighbour_ids.append(movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids

    def find_common_movie_rec(self, movie_set_list: List[set]) -> set:
        if len(movie_set_list) == 1:
            return movie_set_list[0]

        resulting_set = movie_set_list[0]
        for m_set in movie_set_list[1:]:
            resulting_set = resulting_set.intersection(m_set)

        return resulting_set

    def find_group_rec(self, group_movie_list_ids: List[int], k, metric) -> set:
        group_rec = []
        for m_id in group_movie_list_ids:
            output_m_ids = self.find_similar_movies(m_id, self.Q.T, self.movie_mapper, self.movie_inv_mapper, k, metric)
            m_titles = set(self.movie_titles[i] for i in output_m_ids)
            group_rec.append(m_titles)
        group_rec = self.find_common_movie_rec(group_rec)
        return group_rec


start = time.time()

item_item_recommender = ItemitemWithKNNRec()
group_rec = item_item_recommender.find_group_rec([86332, 4993, 192389, 168250, 79132, 91500, 109487], k=1000, metric="cosine")
print(group_rec, end='\n\n')

end = time.time()

print("The time of execution of above program is :", (end-start), "sec")


# toby_ratings_df = pd.DataFrame([{'title': 'avatar-the-way-of-water', 'rating': 3.0}, {'title': 'glass-onion', 'rating': 3.5}, {'title': 'dont-worry-darling', 'rating': 4.5}, {'title': 'nope', 'rating': 4.0}, {'title': 'doctor-strange-in-the-multiverse-of-madness', 'rating': 3.5}, {'title': 'spider-man-no-way-home', 'rating': 4.5}, {'title': 'eternals', 'rating': 2.5}, {'title': 'dune-2021', 'rating': 2.0}, {'title': 'shang-chi-and-the-legend-of-the-ten-rings', 'rating': 4.0}, {'title': 'black-widow-2021', 'rating': 2.5}, {'title': 'loki-2021', 'rating': 3.5}, {'title': 'holidate', 'rating': 2.0}, {'title': 'the-queens-gambit', 'rating': 4.5}, {'title': 'tenet', 'rating': 4.0}, {'title': 'hamilton-2020', 'rating': 4.0}, {'title': 'extraction-2020', 'rating': 2.5}, {'title': 'star-wars-the-rise-of-skywalker', 'rating': 1.5}, {'title': 'knives-out-2019', 'rating': 5.0}, {'title': 'joker-2019', 'rating': 4.5}, {'title': 'ad-astra-2019', 'rating': 3.5}, {'title': 'it-chapter-two', 'rating': 3.0}, {'title': 'avengers-endgame', 'rating': 3.5}, {'title': 'captain-marvel', 'rating': 3.0}, {'title': 'fantastic-beasts-the-crimes-of-grindelwald', 'rating': 2.5}, {'title': 'life-itself-2018', 'rating': 4.0}, {'title': 'free-solo', 'rating': 3.5}, {'title': 'mamma-mia-here-we-go-again', 'rating': 3.0}, {'title': 'oceans-eight', 'rating': 4.0}, {'title': 'blackkklansman', 'rating': 4.5}, {'title': 'solo-a-star-wars-story', 'rating': 4.0}, {'title': 'avengers-infinity-war', 'rating': 4.0}, {'title': 'black-panther', 'rating': 4.5}, {'title': 'star-wars-the-last-jedi', 'rating': 4.0}, {'title': 'jumanji-welcome-to-the-jungle', 'rating': 4.0}, {'title': 'wonder-2017', 'rating': 4.0}, {'title': 'murder-on-the-orient-express-2017', 'rating': 4.0}, {'title': 'coco-2017', 'rating': 4.5}, {'title': 'thor-ragnarok', 'rating': 3.5}, {'title': 'kingsman-the-golden-circle', 'rating': 3.0}, {'title': 'it-2017', 'rating': 4.0}, {'title': 'dunkirk-2017', 'rating': 3.0}, {'title': 'spider-man-homecoming', 'rating': 4.5}, {'title': 'good-time', 'rating': 3.5}, {'title': 'pirates-of-the-caribbean-dead-men-tell-no-tales', 'rating': 3.0}, {'title': 'wonder-woman-2017', 'rating': 3.5}, {'title': 'beauty-and-the-beast-2017', 'rating': 3.0}, {'title': 'passengers-2016', 'rating': 3.0}, {'title': 'rogue-one-a-star-wars-story', 'rating': 4.5}, {'title': 'hidden-figures', 'rating': 4.0}, {'title': 'doctor-strange-2016', 'rating': 4.5}, {'title': 'moana-2016', 'rating': 3.5}, {'title': 'sing-2016', 'rating': 3.5}, {'title': 'lion', 'rating': 4.5}, {'title': 'la-la-land', 'rating': 4.5}, {'title': 'train-to-busan', 'rating': 4.0}, {'title': 'captain-america-civil-war', 'rating': 4.0}, {'title': 'sausage-party', 'rating': 2.0}, {'title': 'hush-2016', 'rating': 3.0}, {'title': 'zootopia', 'rating': 4.0}, {'title': 'deadpool', 'rating': 3.0}, {'title': 'star-wars-the-force-awakens', 'rating': 3.5}, {'title': 'the-intern-2015', 'rating': 3.0}, {'title': 'the-martian', 'rating': 4.5}, {'title': 'ant-man', 'rating': 4.5}, {'title': 'avengers-age-of-ultron', 'rating': 3.5}, {'title': 'the-interview-2014', 'rating': 3.5}, {'title': 'interstellar', 'rating': 4.5}, {'title': 'big-hero-6', 'rating': 4.0}, {'title': 'the-maze-runner', 'rating': 2.5}, {'title': 'maleficent', 'rating': 3.5}, {'title': 'divergent', 'rating': 3.5}, {'title': 'the-hunger-games-catching-fire', 'rating': 3.5}, {'title': 'gravity-2013', 'rating': 3.0}, {'title': 'were-the-millers', 'rating': 4.0}, {'title': 'snowpiercer', 'rating': 3.0}, {'title': 'sharknado', 'rating': 1.5}, {'title': '42', 'rating': 4.0}, {'title': 'lincoln', 'rating': 3.5}, {'title': 'pitch-perfect', 'rating': 3.5}, {'title': 'the-dark-knight-rises', 'rating': 4.0}, {'title': 'moonrise-kingdom', 'rating': 4.5}, {'title': 'the-hunger-games', 'rating': 4.5}, {'title': 'harry-potter-and-the-deathly-hallows-part-2', 'rating': 3.5}, {'title': 'pariah', 'rating': 3.5}, {'title': 'harry-potter-and-the-deathly-hallows-part-1', 'rating': 3.5}, {'title': 'despicable-me', 'rating': 4.0}, {'title': 'inception', 'rating': 4.5}, {'title': 'harry-potter-and-the-half-blood-prince', 'rating': 3.5}, {'title': 'the-hangover', 'rating': 3.0}, {'title': 'up', 'rating': 4.0}, {'title': 'coraline', 'rating': 4.0}, {'title': '35-shots-of-rum', 'rating': 3.0}, {'title': 'the-dark-knight', 'rating': 5.0}, {'title': 'walle', 'rating': 4.5}, {'title': 'kung-fu-panda', 'rating': 4.0}, {'title': 'harry-potter-and-the-order-of-the-phoenix', 'rating': 4.0}, {'title': 'ratatouille', 'rating': 4.0}, {'title': 'superbad', 'rating': 4.0}, {'title': 'inside-man', 'rating': 4.0}, {'title': 'harry-potter-and-the-goblet-of-fire', 'rating': 4.0}, {'title': 'batman-begins', 'rating': 4.0}, {'title': 'star-wars-episode-iii-revenge-of-the-sith', 'rating': 3.5}, {'title': 'the-matrix-revolutions', 'rating': 3.0}, {'title': 'elf', 'rating': 4.0}, {'title': 'school-of-rock', 'rating': 4.0}, {'title': 'finding-nemo', 'rating': 3.5}, {'title': 'the-matrix-reloaded', 'rating': 3.5}, {'title': 'holes', 'rating': 4.0}, {'title': 'harry-potter-and-the-chamber-of-secrets', 'rating': 4.0}, {'title': '8-mile', 'rating': 4.0}, {'title': 'star-wars-episode-ii-attack-of-the-clones', 'rating': 2.5}, {'title': 'oceans-eleven-2001', 'rating': 4.5}, {'title': 'harry-potter-and-the-philosophers-stone', 'rating': 3.5}, {'title': 'spirited-away', 'rating': 3.5}, {'title': 'remember-the-titans', 'rating': 3.5}, {'title': 'chicken-run', 'rating': 3.5}, {'title': 'the-sixth-sense', 'rating': 3.0}, {'title': 'star-wars-episode-i-the-phantom-menace', 'rating': 3.0}, {'title': 'the-matrix', 'rating': 4.5}, {'title': 'mulan', 'rating': 4.0}, {'title': 'the-truman-show', 'rating': 5.0}, {'title': 'good-will-hunting', 'rating': 4.5}, {'title': 'independence-day', 'rating': 3.0}, {'title': 'toy-story', 'rating': 4.0}, {'title': 'quiz-show', 'rating': 4.0}, {'title': 'the-lion-king', 'rating': 4.0}, {'title': 'home-alone-2-lost-in-new-york', 'rating': 3.0}, {'title': 'home-alone', 'rating': 4.0}, {'title': 'back-to-the-future-part-ii', 'rating': 4.0}, {'title': 'die-hard', 'rating': 2.5}, {'title': 'back-to-the-future', 'rating': 4.5}, {'title': 'return-of-the-jedi', 'rating': 3.5}, {'title': 'the-shining', 'rating': 4.0}, {'title': 'straight-time', 'rating': 3.5}, {'title': 'star-wars', 'rating': 4.0}])
# sorted_ratings = toby_ratings_df.sort_values(by="rating", ascending=False)
# print(sorted_ratings.head(30))


























