from PredictionUtils import predict_movie_and_rating,mf
from DiversityUtils import diversifyCandidates

userId= 21
top_20 = predict_movie_and_rating(userId)
kDiversifiedItems = 4
diversifyCandidates(top_20,4,mf.Q)
#print("M main",mf.Q[92])
#print(mf.Q)
# from DiversityUtils import cosdis,word2vec
#
# restaurant = "Restaurant"
# musee = "Musee"
# hotel = "Hotel"
#
# print("res muse",cosdis(word2vec(restaurant),word2vec(musee)))
# print("res hot",cosdis(word2vec(restaurant),word2vec(hotel)))


