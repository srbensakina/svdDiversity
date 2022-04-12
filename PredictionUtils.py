import pandas as pd
import numpy as np
from MatrixFactorization import MF

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


def predict_movie_and_rating(user_id):
    r= mf.full_matrix()
    if user_id in ratings_filtered.user_id.unique():

        # List of movies that the user has seen
        movie_id_list = ratings_filtered[ratings_filtered.user_id == user_id].place_id.tolist()
        # for movie in movie_id_list:
        # a = movie_data.loc[movie_data['movie_id'] == movie]['title'].values.tolist()
        # print(a)" List of movies that the user has not seen
        unseen = {k: v for k, v in name_id_map.items() if not v in movie_id_list}
        liked = [x for x in movie_id_list if x not in unseen.values()]
        # print(liked)
        # print(unseen.values())
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
        return top_10, liked
    # Return the top 10 predicted rated movies

    else:
        print("User Id does not exist in the list!")
        return None
