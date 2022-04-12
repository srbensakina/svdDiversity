from PredictionUtils import predict_movie_and_rating
from DiversityUtils import diversifyCandidates
# userId= 21
# top_20, liked = predict_movie_and_rating(userId)
# kDiversifiedItems = 5
# diversifyCandidates(top_20,5)
#


from scipy import spatial
List1 = [4]
List2 = [3]
result = 1 - spatial.distance.cosine(List1, List2)
print("result :",result)