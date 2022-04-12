from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk


class Diversification:

    def __init__(self, item_data):
        ''' Sets ratings, sim_options, trainset.
            Cleans item_data dataframe, in this case, based on MovieLens dataset.
            Sets items.
        '''

        self.items = item_data
        # self.sim_options = {'name': 'cosine', 'user_based': False}

    def calc_similarity_matrix(self):
        ''' Calculates the items similarity matrix using cosine similarity. This function was developed based on MovieLens dataset, using titles and genres.
        Sets cosine_sim_movies_title, cosine_sim_movies_genres
        '''
        # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        nltk.download('stopwords')
        tfidf = TfidfVectorizer(stop_words=stopwords.words('french'))

        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix_title = tfidf.fit_transform(self.items['title'])
        tfidf_matrix_genres = tfidf.fit_transform(self.items['genre'])
        print(tfidf_matrix_genres)

        # Compute the cosine similarity matrix
        self.cosine_sim_movies_title = cosine_similarity(tfidf_matrix_title, tfidf_matrix_title)
        self.cosine_sim_movies_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)

    def get_similar_items(self, references, liked,title_weight=0.8, k=10):
        ''' Searches for the top-k most similar items in candidate items to a given reference list. This function is based on MovieLens dataset.
            Returns a list of items.
        '''
        recs = []
        for item in references['place_id']:
            # Get the pairwsie similarity scores of all movies with that movie
            movie_idx = item
            sim_scores_title = list(enumerate(self.cosine_sim_movies_title[movie_idx]))
            sim_scores_genres = list(enumerate(self.cosine_sim_movies_genres[movie_idx]))

            # Calculate total similarity based on title and genres
            total_sim_score = []
            for i in range(len(sim_scores_title)):
                aux = (sim_scores_title[i][1] * title_weight) + (sim_scores_genres[i][1] * (1 - title_weight))
                total_sim_score.append((i, aux))

            # Sort the movies based on the similarity scores
            total_sim_score = sorted(total_sim_score, key=lambda x: x[1], reverse=True)

            candidates_sim_score = []
            for sim_item in total_sim_score:
                if self.items.loc[sim_item[0]].values[0] not in liked:
                    candidates_sim_score.append(sim_item)

            # Get the scores of the top-k most similar movies
            k = k + 1
            candidates_sim_score = candidates_sim_score[1:k]
            recs.append(candidates_sim_score)

        return recs

    def get_relevance_score(self, recs, references):
        ''' Calculates the relevance of recommendations.
            Creates a dictionary for better manipulation of data, containing:
                movie_id, movie_title, movie_genres, movie_similarity and movie_relevance. This function is based on MovieLens dataset.
            Returns a dict sorted by movie_relevance.
        '''
        count = 0
        recs_dict = []
        for reference in references['rating']:
            # print('Referência: {}\t gêneros: {}'.format(refinedMyAlgo.movies[refinedMyAlgo.movies[
            # 'movieId']==reference['movieID']].values[0][1], refinedMyAlgo.movies[refinedMyAlgo.movies[
            # 'movieId']==reference['movieID']].values[0][2]))

            for movie in recs[count]:
                aux = {}
                movie_id = self.items.loc[movie[0]].values[0]
                movie_title = self.items.loc[movie[0]].values[1]
                movie_genres = self.items.loc[movie[0]].values[2]
                movie_similarity = movie[1]
                movie_relevance = round(((reference/ 5.0) + movie_similarity) / 2, 3)

                aux['movie_id'] = movie_id
                aux['movie_title'] = movie_title
                aux['movie_genres'] = movie_genres
                aux['movie_similarity'] = movie_similarity
                aux['movie_relevance'] = movie_relevance

                recs_dict.append(aux)

            # print('\tSim: {},\trelevance: {},\tmovieId: {},\ttitle: {}'.format(aux['movie_similarity'],
            # aux['movie_relevance'], aux['movie_id'], aux['movie_title']))

            count = count + 1

        recs_dict = sorted(recs_dict, key=lambda i: i['movie_relevance'], reverse=True)
        return recs_dict

    def calc_distance_item_in_list(self, item, this_list, title_weight=0.8):
        ''' Calculates the total distance of an item in relation to a given list.
            Returns the total distance.
        '''
        idx_i = int(self.items[self.items['place_id'] == int(item['movie_id'])].index[0])

        total_dist = 0
        for movie in this_list:
            idx_j = int(self.items[self.items['place_id'] == int(movie['movie_id'])].index[0])

            sim_i_j = (self.cosine_sim_movies_title[idx_i][idx_j] * title_weight) + (
                        self.cosine_sim_movies_genres[idx_i][idx_j] * (1 - title_weight))
            dist_i_j = 1 - sim_i_j
            total_dist = total_dist + dist_i_j

        result = total_dist / len(this_list)

        return result

    def calc_diversity_score(self, actual_list, candidates_list, alfa=0.5):
        '''
            This function implemented here was based on MARIUS KAMINSKAS and DEREK BRIDGE paper: Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems

                func(i,R) = (relevance[i]*alfa) + (dist_i_R(i,R)*(1-alfa))

            Calculates the diversity score that an item represents to a given list.
            Returns a dict with calculated values.
        '''
        diversity_score = []
        count = 0

        for item in candidates_list:
            aux = {}
            dist_item_R = self.calc_distance_item_in_list(item=item, this_list=actual_list)
            aux['div_score'] = (item['movie_relevance'] * alfa) + (dist_item_R * (1 - alfa))
            aux['idx'] = count
            diversity_score.append(aux)
            count = count + 1

        return diversity_score

    def diversify_recs_list(self, recs, k=10):
        '''
            This function implemented here was based on MARIUS KAMINSKAS and DEREK BRIDGE paper: Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems

                The Greedy Reranking Algorithm.

            Given a list, returns another list with top-k items diversified based on the Greedy algorithm.
        '''
        diversified_list = []

        while len(diversified_list) < k:
            if len(diversified_list) == 0:
                diversified_list.append(recs[0])
                recs.pop(0)
            else:
                diversity_score = self.calc_diversity_score(actual_list=diversified_list, candidates_list=recs)
                diversity_score = sorted(diversity_score, key=lambda i: i['div_score'], reverse=True)
                #  Add the item that maximize diversity in the list
                item = diversity_score[0]
                diversified_list.append(recs[item['idx']])
                #  Remove this item from the candidates list
                recs.pop(item['idx'])
        return diversified_list