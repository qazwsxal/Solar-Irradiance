import ephem
import gpxpy
import gpxpy.gpx
import math
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import random
import pickle
# import pytz


def haversine(GPXPoint1, GPXPoint2):
    earth_rad = 6371*(10**3)
    lat1 = math.radians(GPXPoint1.latitude)
    lat2 = math.radians(GPXPoint2.latitude)
    lon1 = math.radians(GPXPoint1.longitude)
    lon2 = math.radians(GPXPoint2.longitude)
    delta_lat = (lat2-lat1)
    delta_lon = (lon2-lon1)
    a = math.pow(math.sin(delta_lat/2), 2) + \
        math.cos(lat1) * math.cos(lat2) * \
        math.pow(math.sin(delta_lon/2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return earth_rad * c


def forward_azimuth(GPXPoint1, GPXPoint2):
    lat1 = math.radians(GPXPoint1.latitude)
    lat2 = math.radians(GPXPoint2.latitude)
    lon1 = math.radians(GPXPoint1.longitude)
    lon2 = math.radians(GPXPoint2.longitude)
    delta_lon = (lon2-lon1)
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - \
        math.sin(lat1)*math.cos(lat2)*math.cos(delta_lon)
    return math.degrees(math.atan2(y, x))


def random_time(start, end):
    timerange = end - start
    addtime = timerange.total_seconds()*random.random()
    delta = dt.timedelta(seconds=addtime)
    return start + delta

gpx_file = open('wsc.gpx', 'r')


# PyEphem needs UTC times
START = dt.datetime(2017, 10, 10, 22, 30)
END = dt.datetime(2017, 10, 11, 7, 30)

gpx = gpxpy.parse(gpx_file)
path = gpx.tracks[0].segments[0].points
# for track in gpx.tracks:
#     for segment in track.segments:
#         for point in segment.points:
#             print('Point at ({0},{1}) -> {2}'.format(point.latitude,
#                                                      point.longitude,
#                                                      point.elevation))
lats = []
lons = []
elvs = []
bearings = []
dists = []
for i in range(len(path)-1):
    lats.append(path[i].latitude)
    lons.append(path[i].longitude)
    elvs.append(path[i].elevation)
    bearings.append(forward_azimuth(path[i], path[i+1]))
    dists.append(haversine(path[i], path[i+1]))


# journ = np.cumsum(dists)
# plt.plot(journ, elvs)
# plt.show()
# plt.hist(dists, bins=100)
# print(max(dists),np.argmax(dists))
# print(min(dists),np.argmin(dists))
# plt.show()
bins = 500
histdata = np.histogram(bearings, bins=bins, weights=dists)
count = histdata[0]
angles = np.radians(histdata[1])
bottom = max(count)*(2/3)
width = (2*np.pi) / len(count)

ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
bars = ax.bar(angles[:-1], count, width=width, bottom=bottom)
plt.show()
alts = []
azs = []
weights = []
car_position = ephem.Observer()
for i in range(len(dists)):
    car_position.lon = str(path[i].longitude)
    car_position.lat = str(path[i].latitude)
    car_position.elevation = path[i].elevation
    if (i % 100 == 0):
        print(100*i/len(dists), '%')
    for j in range(100):
        car_position.date = random_time(START, END)
        sun = ephem.Sun(car_position)
        alts.append(sun.alt)
        azs.append((sun.az - math.radians(bearings[i])) % 2*math.pi)
        weights.append(min(dists[i], 5000))

alts.append(-math.pi)
azs.append(math.pi)
weights.append(0)
alts.append(math.pi)
azs.append(math.pi)
weights.append(0)
plt.hist2d(azs, alts, bins=256, weights=weights)
results = np.histogram2d(azs, alts, bins=4096)
pickle.dump(results, open("results.pkl", "wb"))
plt.show()
