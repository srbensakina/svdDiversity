import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy import spatial
from scipy.spatial import distance

places_data = pd.io.parsers.read_csv('data/places.csv', names=['place_id', 'title', 'genre'], engine='python',
                                     delimiter=',')


def diversifyCandidates(candidates, k, Q):
    diversifiedList = pd.DataFrame([candidates.iloc[0]])
    candidates = candidates.drop(0)
    candidates = candidates.reset_index(drop=True)
    print("candidates length : ", len(candidates))
    print("diversified length : ", len(diversifiedList))

    print("************ diversified List : ************")

    while len(diversifiedList) != k:
        arrayOfDistance = []
        for i in range(len(candidates)):
            DistanceOfCandidatesToDiversified = []
            for j in range(len(diversifiedList)):
                print("******************* iteration i : ", i, "iteration  j :", j, "**********************")
                idC = candidates.iloc[i]['place_id']
                idD = diversifiedList.iloc[j]['place_id']

                vectorC = Q[idC]
                vectorD = Q[idD]
                DistanceOfCandidatesToDiversified.append(spatial.distance.euclidean(vectorC , vectorD))

            print("M each array candidat", DistanceOfCandidatesToDiversified)
            arrayOfDistance.append(sum(DistanceOfCandidatesToDiversified) / len(diversifiedList))
        print("M the final array", arrayOfDistance)
        maxDistanceIndex = (np.where(arrayOfDistance == np.amax(arrayOfDistance))[0])[0]
        print("min cosine :", maxDistanceIndex)
        print("list of candidates :", candidates)
        print("object to append", candidates.iloc[maxDistanceIndex])
        diversifiedList = diversifiedList.append(candidates.iloc[maxDistanceIndex], ignore_index=True)
        print("diversified ", diversifiedList, "length : ", len(diversifiedList))
        candidates = candidates.drop(maxDistanceIndex)
        candidates = candidates.reset_index(drop=True)
