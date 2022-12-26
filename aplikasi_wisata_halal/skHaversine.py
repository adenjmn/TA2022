from math import *
import pandas as pd
import numpy as np
# import math
import sqlite3

import geocoder
g = geocoder.ip('me')
curr_lat = g.lat
curr_lon = g.lng

conn = sqlite3.connect('tugas-akhir.db')
cursor = conn.cursor()

sql_maps = "SELECT tempat,city,coo1,coo2 FROM data"
cursor.execute(sql_maps)
result_maps = cursor.fetchall()



def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2 - lat1)
   delta_lambda = np.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
   return np.round(res, 2)

start_lat, start_lon = curr_lat, curr_lon

haversine = pd.DataFrame(result_maps,columns= ['tempat','City','Lat','Lon'])
# haversine = haversine.dropna()
distances_km = []
for row in haversine.itertuples(index=False):
   distances_km.append(
       haversine_distance(start_lat, start_lon, row.Lat, row.Lon)
   )

haversine['Distance'] = distances_km
haversine = haversine.sort_values(by='Distance', ascending=True)
print(haversine.head())
print(haversine.info())

sql_meta = "SELECT id_tempat,jenis, coordinat, tempat,city,img, description FROM data"
cursor.execute(sql_meta)
result_meta = cursor.fetchall()
meta = pd.DataFrame(result_meta,columns= ['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi'])
meta = haversine.merge(meta[['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi']],how="left", on=["tempat"])
print(meta.head())
print(meta.info())