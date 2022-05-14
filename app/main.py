from pickletools import long1
from typing import List, Optional

from app.PredictionUtils import predict_movie_and_rating, mf
from app.DiversityUtils import diversifyCandidates
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel


class Place(BaseModel):
    name: str
    id: int

    def to_Place_Row(self):
        return "\n{},{}".format(self.id, self.name)


class Rate(BaseModel):
    user_id: str
    place_id: int
    rating: int

    def to_Rate_Row(self):
        return "\n{},{},{}".format(self.user_id, self.place_id, self.rating)


app = FastAPI()


@app.get("/")
def read_root():
    return {"Recommending": "Api"}


@app.post("/api/v1/update/places")
def update_places(place: Place):
    with open('data/places.csv', 'a') as places:
        places.write(place.to_Place_Row())
    return "success"


@app.post("/api/v1/update/ratings")
def update_ratings(rate: Rate):
    with open('data/placesratings.csv', 'a') as placesRatings:
        placesRatings.write(rate.to_Rate_Row)
    return "success"


@app.get("/api/v1/users/{user_id}/places/recommended")
def get_diversified_recommendations(user_id: int):
    top_20, l = predict_movie_and_rating(user_id)
    print("hmmm ", top_20.loc[:, 'place_id'].values)
    return JSONResponse(content=top_20.loc[:, 'place_id'].tolist())


@app.get("/api/v1/users/{user_id}/places/diversified")
def get_diversified_recommendations(user_id: int):
    top_20, l = predict_movie_and_rating(user_id)
    kDiversifiedItems = 4
    data = diversifyCandidates(top_20, kDiversifiedItems, mf.Q, l)
    return JSONResponse(content=data.loc[:, 'place_id'].tolist())
