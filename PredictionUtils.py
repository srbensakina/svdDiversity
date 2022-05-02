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
#Rdf = pd.DataFrame(R, index=None, columns=None)
#list1 = []
#list2 = []
#lis3 = []
#for data in Rdf.iteritems():
   # list1.append(np.count_nonzero(data[1]))
   # list2.append(np.mean(data[1]))
   # print(list2)
   # lis3.append(data[1])
#Rdf.loc['v'] = list1
#m = np.mean(Rdf.loc['v'])
#for place in list1:
   # Rdf.loc['w'] = Rdf.loc['v'] / (Rdf.loc['v'] + m)
#Rdf.loc['R'] = list2
#C = np.nanmean(lis3)
#popularity = np.array(Rdf.loc['w'] * Rdf.loc['R'] + (1 - Rdf.loc['w']) * C)
# R['v'] = R[user_line].count(axis=1)
mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=5)
training_process = mf.train()
# np.savetxt('factomatrix.csv', mf.full_matrix(), delimiter=',')
# np.savetxt('Q.csv', mf.Q, delimiter=',')



#   start_time = time.time()
#  np.savetxt("data/mfcache.csv", mf.full_matrix(), delimiter=",")
# print("--- %s seconds ---" % (time.time() - start_time))

# print()
# print("P x Q:")
# a = mf.full_matrix()
# print(mf.full_matrix())
# print()


def predict_movie_and_rating(user_id):
    r = mf.full_matrix()
    if user_id in ratings_filtered.user_id.unique():

        # List of movies that the user has seen
        movie_id_list = ratings_filtered[ratings_filtered.user_id == user_id].place_id.tolist()
        unseen = {k: v for k, v in name_id_map.items() if not v in movie_id_list}

        scores = []
        # For each movie that is unseen use the SVD model to predict that movie's rating
        for value in unseen.values():

            index = user_id  # user_id starts with 0

            predicted = r.item(index, value)

            title = movies_filtered.iloc[value]['title']
            movie_prediction = [value, predicted, title]
            scores.append(movie_prediction)
        # Make a dataframe for the recommendations
        recommendations = pd.DataFrame(scores, columns=['place_id', 'rating', 'title'])
        recommendations.sort_values('rating', ascending=False,
                                    inplace=True)
        recommendations.index = range(len(scores))

        top_10 = recommendations.head(8)

        print("\n\n--> We commend")
        print(top_10)
        # print(id)
        return top_10
    # Return the top 10 predicted rated movies

    else:
        print("User Id does not exist in the list!")
        return None
