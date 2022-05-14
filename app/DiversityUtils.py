import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy import spatial
from scipy.spatial import distance

places_data = pd.io.parsers.read_csv('data/places.csv', names=['place_id', 'title', 'genre'], engine='python',
                                     delimiter=',')


def diversifyCandidates(candidates, k, Q, l):
    diversifiedList = pd.DataFrame([candidates.iloc[0]])
    # candidates = candidates.drop(0)
    # candidates = candidates.reset_index(drop=True)
    print("candidates length : ", len(candidates))
    print("diversified length : ", len(diversifiedList))

    print("************ diversified List : ************")

    while len(diversifiedList) != k:
        arrayOfDistance = []
        for i in range(len(l)):
            DistanceOfCandidatesToDiversified = []
            for j in range(len(diversifiedList)):
                print("******************* iteration i : ", i,
                      "iteration  j :", j, "**********************")
                idC = l.iloc[i]['place_id']
                idD = diversifiedList.iloc[j]['place_id']

                vectorC = Q[idC]
                vectorD = Q[idD]
                DistanceOfCandidatesToDiversified.append(
                    spatial.distance.euclidean(vectorC, vectorD))

            print("M each array candidat", DistanceOfCandidatesToDiversified)
            arrayOfDistance.append(
                sum(DistanceOfCandidatesToDiversified) / len(diversifiedList))
        print("M the final array", arrayOfDistance)
        maxDistanceIndex = (
            np.where(arrayOfDistance == np.amax(arrayOfDistance))[0])[0]
        print("min cosine :", maxDistanceIndex)
        print("list of candidates :", candidates)
        print("object to append", l.iloc[maxDistanceIndex])
        diversifiedList = diversifiedList.append(
            l.iloc[maxDistanceIndex], ignore_index=True)
        print("diversified ", diversifiedList,
              "length : ", len(diversifiedList))
        l = l.drop(maxDistanceIndex)
        l = l.reset_index(drop=True)
    return diversifiedList.loc[:, 'place_id'].to_json()
