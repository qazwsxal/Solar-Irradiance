import math
import random
import datetime as dt
import csv

import ephem
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import numpy as np
# import pytz


def haversine(gpxpoint1, gpxpoint2):
    ''' Returns distance between two points on a sphere

    Returns great-circle distance between two points, takes two objects
    with "latitude" and "longitude" attributes in degrees
    '''
    earth_rad = 6371*(10**3)
    lat1 = math.radians(gpxpoint1.latitude)
    lat2 = math.radians(gpxpoint2.latitude)
    lon1 = math.radians(gpxpoint1.longitude)
    lon2 = math.radians(gpxpoint2.longitude)
    delta_lat = (lat2-lat1)
    delta_lon = (lon2-lon1)
    a = math.pow(math.sin(delta_lat/2), 2) + \
        math.cos(lat1) * math.cos(lat2) * \
        math.pow(math.sin(delta_lon/2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return earth_rad * c


def forward_azimuth(gpxpoint1, gpxpoint2):
    lat1 = math.radians(gpxpoint1.latitude)
    lat2 = math.radians(gpxpoint2.latitude)
    lon1 = math.radians(gpxpoint1.longitude)
    lon2 = math.radians(gpxpoint2.longitude)
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


def altaz2vec(altitude, azimuth):
    z = math.sin(altitude)
    h = math.cos(altitude)
    x = h*math.sin(azimuth)
    y = h*math.cos(azimuth)
    return [x, y, z]



panel_curve = []
with open('SolarCarBody_Pressure.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        panel_curve.append(list(map(lambda x: float(x.strip('m')), row))[0:2])
panel_curve = np.array(panel_curve)
panel_grads = np.diff(panel_curve, axis=0)
panel_norms = np.transpose(np.dot(np.array([[0, 1],[0, 0], [-1, 0]]), np.transpose(panel_grads)))
panel_norms = panel_norms/np.linalg.norm(panel_norms, axis=1)[:, None]
print(panel_norms)
gpx_file = open('wsc.gpx', 'r')


# PyEphem needs UTC times
START = dt.datetime(2017, 10, 10, 22, 30)
END = dt.datetime(2017, 10, 11, 7, 30)

# print(np.sum(panvects, axis=1))

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


# plt.plot(journ, elvs)
# plt.show()
# plt.hist(dists, bins=100)
# plt.show()
alts = []
azs = []
car_position = ephem.Observer()
print("Calculating Sun Positions")
for i in range(len(dists)):
    car_position.lon = str(path[i].longitude)
    car_position.lat = str(path[i].latitude)
    car_position.elevation = path[i].elevation
    if i % 100 == 0:
        print(100*i/len(dists), '%')
    for j in range(100):
        car_position.date = random_time(START, END)
        sun = ephem.Sun(car_position)
        alts.append(sun.alt)
        # Work out sun position relative to car
        azs.append((sun.az - math.radians(bearings[i])) % 2*math.pi)

print("Determing Sun Vectors")
sunvects = np.transpose(np.array(list(map(altaz2vec, alts, azs))))
print("Calculating Irradiances")
irrad = np.dot(panel_norms, sunvects)
result = np.sum(irrad, axis=1)/sunvects.shape[1]

plane = np.array([[0,0,1],[0,0,1]])
plane_dims = np.array([panel_curve[-2,0], panel_curve[0,0]])
plane_irrad = np.dot(plane,sunvects)
plane_result = np.sum(plane_irrad, axis=1)/sunvects.shape[1]
# plt.plot(pandists, result)
plt.subplot(211)
# plt.plot(panel_curve[:-1,0],result/np.linalg.norm(result, ord=np.inf))
plt.plot(panel_curve[:-1,0],result)
plt.plot(plane_dims,plane_result)
plt.ylim(0.7,0.8)
plt.subplot(212, aspect='equal')
plt.plot(panel_curve[:-1,0],panel_curve[:-1,1])
plt.ylim(-20,220)
# plt.axis('equal')
# plt.plot(pandists/np.linalg.norm(pandists, ord=np.inf))
plt.show()
