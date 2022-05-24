import logging
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from asyncio.log import logger
import time
from apscheduler.schedulers.background import BackgroundScheduler


from typing import List
from app.PredictionUtils import predict_movie_and_rating, mf
from app.DiversityUtils import diversifyCandidates
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel


def userToInteger(userid: str):
    ratings_data = pd.io.parsers.read_csv('data/userToInteger.csv', names=['user_str', 'user_int'],
                                          engine='python', delimiter=',')
    userint = ratings_data.loc[ratings_data['user_str']
                               == userid].loc[:, 'user_int'].values[0]
    print("user int is ", userint)
    return userint


class Place(BaseModel):
    name: str
    id: int

    def to_Place_Row(self):
        return "\n{},{}".format(self.id, self.name)


class Location:
    coordinates: List[int]
    type: str

    def __init__(self):
        self.coordinates = [0, 0]
        self.type = "Point"


class BigPlace(BaseModel):
    id: int
    name: str
    description: str
    type: str
    imglink: str
    location: Location

    class Config:
        arbitrary_types_allowed = True


class Rate(BaseModel):
    user_id: str
    place_id: int
    rating: int

    def to_Rate_Row(self):
        return "\n{},{},{}".format(self.user_id, self.place_id, self.rating)


app = FastAPI()


# APScheduler Related Libraries


Schedule = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def load_schedule_or_create_blank():
    """
    Instatialise the Schedule Object as a Global Param and also load existing Schedules from SQLite
    This allows for persistent schedules across server restarts. 
    """
    global Schedule
    try:
        jobstores = {
            'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
        }
        Schedule = AsyncIOScheduler(jobstores=jobstores)
        Schedule.start()

        logger.info("Created Schedule Object")
    except:
        logger.error("Unable to Create Schedule Object")


@app.get("/")
def read_root():
    return {"Recommending": "Api"}


def training_job():
    print("started")
    mf.train()


@app.get("/api/v1/users/start_training")
def start_training():
    Schedule.add_job(training_job, 'interval', hours=24)
    training_job()
    return "done"


@app.get("/placesAsJson")
def get_places_as_json():
    data = pd.io.parsers.read_csv('data/places.csv', names=['place_id', 'title', 'genre'], engine='python',
                                  delimiter=',')
    mylist = []
    for index, row in data.iterrows():
        aplace = BigPlace(id=row['place_id'],
                          name=row['title'], type=row['genre'], description="some wonderful place", imglink="https://res.cloudinary.com/wavy/image/upload/v1577177858/FB_IMG_1576505116453.jpg", location=Location())
        mylist.append(aplace)
    print(Location())
    return mylist


@app.post("/api/v1/update/places")
def update_places(place: Place):
    with open('data/places.csv', 'a') as places:
        places.write(place.to_Place_Row())
    return "success"


@app.post("/api/v1/update/ratings")
def update_ratings(rate: Rate):
    x = 0

    try:
        x = userToInteger(rate.user_id)
    except:
        with open('data/userToInteger.csv', 'r') as userToInteger:
            x = len(userToInteger.readlines())+1
        with open('data/userToInteger.csv', 'a') as userToInteger:
            userToInteger.write("\n{},{}".format(rate.user_id, x))
    rate.user_id = x
    with open('data/ratings.csv', 'a') as placesRatings:
        placesRatings.write(rate.to_Rate_Row())
    return "success"


@app.get("/api/v1/users/{user_id}/places/recommended")
def get_recommended_places(user_id: str):
    top_20, l = predict_movie_and_rating(userToInteger(user_id))
    print("hmmm ", top_20.loc[:, 'place_id'].values)
    return JSONResponse(content=top_20.loc[:, 'place_id'].tolist())


@app.get("/api/v1/users/{user_id}/places/diversified")
def get_diversified_Places(user_id: str):
    top_20, l = predict_movie_and_rating(userToInteger(user_id))
    kDiversifiedItems = 4
    data = diversifyCandidates(top_20, kDiversifiedItems, mf.Q, l)
    return JSONResponse(content=data.loc[:, 'place_id'].tolist())
