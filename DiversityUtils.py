import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy import spatial


places_data = pd.io.parsers.read_csv('data/places.csv', names=['place_id', 'title', 'genre'], engine='python',
                                     delimiter=',')

categoriesAsNumbers = {}

def calculateCos(candidates,diversified):
    return 1 - spatial.distance.cosine(candidates, diversified)
def calculateCosine(candidates, diversified):
    if (len(diversified) < len(candidates)):
        for i in range(len(candidates) - len(diversified)):
            diversified.append(0)
    if (len(diversified) > len(candidates)):
        for i in range(len(candidates) - len(diversified)):
            candidates.append(0)
    print("size of candidates", len(candidates), " diversified:", len(diversified))
    return np.dot(candidates, diversified) / (norm(candidates) * norm(diversified))


def createCategoriesAsNumbersMap():
    k = 1
    for genre in places_data.loc[:, "genre"].to_numpy():
        if genre not in categoriesAsNumbers:
            categoriesAsNumbers[genre] = k
            k = k + 1
    print("***** categories as numbers ***** \n", categoriesAsNumbers)


def generateArrayFromListOfObjects(listOfObjects):
    print("list of objects : ", listOfObjects)
    array = []
    for i in range(len(listOfObjects)):
        # array.append(categoriesAsNumbers[places_data.loc[listOfObjects.loc[i, 'place_id'], 'genre']])
        # print("i is : ",i)
        # print("place id of my item", listOfObjects)
        # print("genre :", places_data.loc[listOfObjects['place_id'].loc[listOfObjects.index[i]], 'genre'])

        array.append(
            categoriesAsNumbers[places_data.loc[listOfObjects['place_id'].loc[listOfObjects.index[i]], 'genre']])
    print("*** generated array is ***:\n", array, "done")
    return array


def diversifyCandidates(candidates, k):
    # print(movie_data[movie_data['place_id'] == candidates.at[0, 'place_id']])

    diversifiedList = pd.DataFrame([candidates.iloc[0]])

    print("candidates length : ", len(candidates))
    print("diversified length : ", len(diversifiedList))

    print("************ diversified List : ************")
    createCategoriesAsNumbersMap()
    #
    # diversifiedAsNp = generateArrayFromListOfObjects(diversifiedList)
    # candidatesAsNp = generateArrayFromListOfObjects(candidates)
    # print("cosine is " ,calculateCosine(candidatesAsNp, diversifiedAsNp))
    copyOfCandidates = pd.DataFrame(candidates)
    copyOfDiversified = pd.DataFrame(diversifiedList)

    arrayOfCosines = []
    # while (len(diversifiedList) != k):
    for i in range(len(candidates)):
        print("******************* iteration ", i, "**********************")
        copyC = pd.DataFrame([candidates.iloc[i]])
        copyD = pd.DataFrame(diversifiedList)
        print("copyD : ",copyD,"copyC : ",copyC)

        diversifiedAsNp = generateArrayFromListOfObjects(copyD)
        candidatesAsNp = generateArrayFromListOfObjects(copyC)
        print("diversifiedAsNp : ",diversifiedAsNp,"candidatesASNp : ",candidatesAsNp)
        arrayOfCosines.append(calculateCosine(candidatesAsNp, diversifiedAsNp))

    print(arrayOfCosines)
