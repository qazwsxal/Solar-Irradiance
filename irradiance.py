import ephem
import gpxpy
import gpxpy.gpx
import math
import numpy as np
import matplotlib.pyplot as plt


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

gpx_file = open('wsc.gpx', 'r')

gpx = gpxpy.parse(gpx_file)

path = gpx.tracks[0].segments[0].points

# for track in gpx.tracks:
#     for segment in track.segments:
#         for point in segment.points:
#             print('Point at ({0},{1}) -> {2}'.format(point.latitude,
#                                                      point.longitude,
#                                                      point.elevation))
bearings = []

for i in range(len(path)-1):
    bearings.append(forward_azimuth(path[i], path[i+1]))

histdata = np.histogram(bearings, bins='auto')
count = histdata[0]
angles = np.radians(histdata[1])
N = 80
bottom = max(count)*(2/3)
width = (2*np.pi) / len(count)

ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)


bars = ax.bar(angles[:-1], count, width=width, bottom=bottom)

# Use custom colors and opacity
plt.show()
