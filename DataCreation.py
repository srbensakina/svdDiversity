import csv
import pandas as pd
import numpy as np
from csv import reader

data = pd.read_csv("data/placesratings.csv")
# print(data)
name = []
category = []
ids = []
for col in data.columns:
    if (col == "Nom/Places"):
        continue
    if (col.split()[0] == "Centre"):
        if (col.split()[1] == "de"):
            full_name = col.split()[0] + " " + col.split()[1] + " " + col.split()[2]
            category.append(full_name)
        else:
            full_name = col.split()[0] + " " + col.split()[1]
            category.append(full_name)
    else:
        category.append(col.split()[0])
    # print(col)
    name.append(col)
for id in range(len(data.columns) - 1):
    ids.append(id)
idspd = np.array(ids)
namepd = np.array(name)
categorypd = np.array(category)
# print(len(idspd),len(categorypd),len(namepd))
df = pd.DataFrame({"id": idspd, "name": namepd, "category": categorypd})
# df.to_csv('places.csv', index=False)

# skip first line i.e. read header first and then iterate over each row od csv as a list
user_id = []
places_id = []
ratings = []
with open('data/placesratings.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            for index, cell in enumerate(row):
                # [1:]
                if cell.isdigit() and int(cell) != 0 :
                    #print(index-1)
                    ratings.append(cell)
                    user_id.append(csv_reader.line_num - 2)
                    places_id.append(index-1)
                else:
                    continue
                if index == 100:
                   continue
user_iddp = np.array(user_id)
places_idpd = np.array(places_id)
ratingspd = np.array(ratings)
print(len(ratings), len(user_id), len(places_id))
# print(len(idspd),len(categorypd),len(namepd))
df_rating = pd.DataFrame({"user_id": user_iddp, "places_id": places_idpd, "rating": ratingspd})
#print(df_rating)
# for column in df_rating['rating']:
# index_names = df_rating[df_rating['rating'] == 0].index
# df_rating.drop(index_names, inplace=True)
# print(df_rating)
df_rating.to_csv('ratings.csv', index=False)
