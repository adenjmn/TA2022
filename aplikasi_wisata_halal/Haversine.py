from math import *
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

center_point = [{'lat': 14.0952829, 'lng': 120.9355611}]
test_point = [{'lat': -6.9222, 'lng': 107.6069}]

lat1 = center_point[0]['lat']
lon1 = center_point[0]['lng']
lat2 = test_point[0]['lat']
lon2 = test_point[0]['lng']

a = haversine(lon1, lat1, lon2, lat2)

print('Distance (km) : ', a)

area = 1.00 # in kilometer
if a <= area:
    print('Inside the area')
else:
    print('Outside the area')