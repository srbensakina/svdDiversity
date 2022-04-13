from PredictionUtils import predict_movie_and_rating
from DiversityUtils import diversifyCandidates
userId= 21
top_20, liked = predict_movie_and_rating(userId)
kDiversifiedItems = 5
diversifyCandidates(top_20,5)


