from PredictionUtils import predict_movie_and_rating,mf
from DiversityUtils import diversifyCandidates

userId = 21
top_20, l = predict_movie_and_rating(userId)
kDiversifiedItems = 4
diversifyCandidates(top_20, 4, mf.Q, l)



