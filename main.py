import numpy as np
from surprise import accuracy
import pandas as pd
from Diversification import Diversification


class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : α learning rate
        - beta (float)  : λ regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


movie_data = pd.io.parsers.read_csv('data/places.csv', names=['place_id', 'title', 'genre'], engine='python',
                                    delimiter=',')

ratings_data = pd.io.parsers.read_csv('data/ratings.csv', names=['user_id', 'place_id', 'rating'],
                                      engine='python', delimiter=',')

movie_data_merged = pd.merge(ratings_data, movie_data, on='place_id')
movie_data_merged.groupby('title')['rating'].count().sort_values(ascending=False)
# print("We have {} unique movies in movies dataset.".format(len(movie_data.movie_id.unique())))

ratings_filtered = ratings_data.groupby('user_id').filter(lambda x: len(x) >= 20)

filtered_movie_ids = ratings_filtered.place_id.unique().tolist()
movies_filtered = movie_data[movie_data.place_id.isin(filtered_movie_ids)]

# Create a dictionary mapping movie names to id's
name_id_map = dict(zip(movies_filtered.title.tolist(), movies_filtered.place_id.tolist()))

ratings_right_filtered = pd.merge(
    movie_data[['place_id']],
    ratings_filtered,
    on="place_id",
    how="right"
)

R = np.array(ratings_right_filtered.pivot(
    index='user_id',
    columns='place_id',
    values='rating'
).fillna(0))
mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=5)
training_process = mf.train()


# try:
#   mfcache = pd.read_csv("data/mfcache.csv")
# except:
#   start_time = time.time()
#  np.savetxt("data/mfcache.csv", mf.full_matrix(), delimiter=",")
# print("--- %s seconds ---" % (time.time() - start_time))


# print()
# print("P x Q:")
# a = mf.full_matrix()
# print(mf.full_matrix())
# print()


def predict_movie_and_rating(user_id, r):
    if user_id in ratings_filtered.user_id.unique():

        # List of movies that the user has seen
        movie_id_list = ratings_filtered[ratings_filtered.user_id == user_id].place_id.tolist()
        # for movie in movie_id_list:
        # a = movie_data.loc[movie_data['movie_id'] == movie]['title'].values.tolist()
        # print(a)" List of movies that the user has not seen
        unseen = {k: v for k, v in name_id_map.items() if not v in movie_id_list}
        liked = [x for x in movie_id_list if x not in unseen.values()]
        #print(liked)
        #print(unseen.values())
        scores = []
        # For each movie that is unseen use the SVD model to predict that movie's rating
        for value in unseen.values():
            # for value in range(len(movies_filtered)):
            # print(value)
            index = user_id  # user_id starts with 0
            # print(a)
            # if value in unseen.values():
            # continue
            predicted = r.item(index, value)
            # predicted = r[user_id][unseen.get(movie_id)]
            # print(movie_id, predicted)
            # scores[movie_id] = predicted[3]
            title = movies_filtered.iloc[value]['title']
            movie_prediction = [value, predicted, title]
            scores.append(movie_prediction)
        # Make a dataframe for the recommendations
        recommendations = pd.DataFrame(scores, columns=['place_id', 'rating', 'title'])
        recommendations.sort_values('rating', ascending=False,
                                    inplace=True)
        recommendations.index = range(len(scores))
        # id = []
        # id= recommendations['place_id'].copy()
        # Sort by decreasing scores as we want higher rated movies at the top
        # recommendations.set_index('movies', inplace=True)
        top_10 = recommendations.head(20)
        # movies_recommended = []
        # print(top_10.iloc[:,[0]].values)
        # for i in top_10.iloc[:, [0]].values:
        # print("i is ",i)
        # movies_recommended = np.append(movies_recommended, movies_filtered.iloc[i]['title'].values)
        print("\n\n--> We commend")
        print(top_10)
        # print(id)
        return top_10,liked
    # Return the top 10 predicted rated movies

    else:
        print("User Id does not exist in the list!")
        return None


user_id = 21
r = mf.full_matrix()
top_10,liked = predict_movie_and_rating(user_id, mf.full_matrix())
#top_10.drop(['title'], axis='columns', inplace=True)
# accuracy.rmse(top_10)
# recs = diversify.get_similar_items(self=diversify,references=top_10)
# candidates_list = diversify.get_relevance_score(self=diversify,recs=recs, references=top_10)
diverse = Diversification(movie_data)
diverse.calc_similarity_matrix()
recs = diverse.get_similar_items(top_10,liked)
print(recs)
candidates_list = diverse.get_relevance_score(recs=recs,references=top_10)
print("\n\n-->  The top-20 STANDARD recs are:\n")
for item in candidates_list[0:20]:
        print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))
my_candidates = candidates_list.copy()
final_recs_greedy = diverse.diversify_recs_list(recs=my_candidates)
print("\n\n-->  The top-10 GREEDY DIVERSIFIED recs are:\n")
for item in final_recs_greedy:
    print('movieId: {}, relevance: {}, title:{}'.format(item['movie_id'], item['movie_relevance'], item['movie_title']))
