
"""


DOCS:
  https://ocefpaf.github.io/python4oceanographers/blog/2015/08/03/fiona_gpx/
"""
from datetime import datetime
import os
import re

import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin

import fiona
import xml.etree.ElementTree as ET

from shapely.geometry import Point


hubs_shp_file = os.path.join(os.getcwd(),"data","hubs.shp")
system_shp_file = os.path.join(os.getcwd(), "data", "system.shp")

routes_directory = os.path.join(os.getcwd(), "routes")


def convert_time(time_string):
  """
  """

  # 
  pass


def get_region(point, mesh, crs):
  """
  point
  """
  point = Point(point[0], point[1])
  series = gpd.GeoSeries(point, crs=crs, name = "breadcrumb")
  df = gpd.GeoDataFrame().append(series)
  df = df.rename(columns={0: 'geometry'}).set_geometry('geometry')

  region = gpd.sjoin(df, mesh, how="inner", op='intersects')

  if len(region) > 0:
    return region.index_right[0]
  else:
    return -1


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians.
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Apply Haversine formula.
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    
    # Radius of earth in kilometers is 6371, convert to meters and return.
    meters = 6371000 * c
    return meters


def gpx2df(gpx_file, hubs, system):
  """
  Returns a pandas dataframe representing a gpx track.
  """
  # Get the name from the file.
  name = gpx_file.split('.')[0]
  route_id = int(re.search("[0-9]+", name).group(0))

  # Read in the gpx file.
  gpx_path = os.path.join(os.getcwd(), "routes", gpx_file)
  tree = ET.parse(gpx_path)

  # Store the root of the XML tree.
  root = tree.getroot()

  # Retrieve the start time of the route.
  dt = root[0].text
  dt = re.sub("-0[45]{1}:00", "", dt)
  dt_obj = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S')
  start_time = int(dt_obj.strftime('%H'))
  weekday = dt_obj.weekday()

  # Get the track segment from the root.
  track_segment = root[2][0]

  # Init df to hold cleaned data.
  ride_df = pd.DataFrame()
  crs = {'init': 'epsg:4326'}

  # Get start hub and end hub.
  start_region = get_region((float(track_segment[0].get('lon')),
                             float(track_segment[0].get('lat'))),
                            hubs, crs)

  end_region = get_region((float(track_segment[len(track_segment)-1].get('lon')),
                           float(track_segment[len(track_segment)-1].get('lat'))),
                          hubs, crs)

  # Init counters and pointers.
  previous = None
  previous_time_region = None
  previous_physical_region = None
  heading = None
  displacement = 0

  # Loop through all points in the track segment.
  for count, breadcrumb in enumerate(track_segment):

    # Get current region.
    bc_pts = (float(breadcrumb.get('lon')), float(breadcrumb.get('lat')))
    current_region = get_region(bc_pts, system, crs)

    # Update previous region only if it is different.
    if current_region != previous_time_region:
      previous_physical_region = previous_time_region

    # Get displacement.
    if count > 0:
      displacement = haversine(prev_pts[0], prev_pts[1],
                               bc_pts[0], bc_pts[1])

    # Update heading only if rider has moved.
    if count > 0 and displacement > 20:
      heading = calculate_initial_compass_bearing(prev_pts, bc_pts)

    # Init Pandas Series
    pseries = pd.Series(data = [gpx_file, route_id, start_time, weekday, start_region,
                                end_region, count, previous_physical_region,
                                current_region, heading, displacement],
                        index = ['gpx file', 'route id', 'start_time', 'day_of_week',
                                 'start_region', 'end_region', 'ride_time',
                                 'previous_region', 'current_region',
                                 'heading', 'displacement'])

    # Append Ride to Dataframe  
    ride_df = ride_df.append(pseries, ignore_index = True)

    # Update previous breadcrumb
    prev_pts = bc_pts

    # Update pointer to last region.
    previous_time_region = current_region

  # Exit
  return ride_df


def build_X(routes_dir):

  routes = [i for i in os.listdir(routes_dir) if i.startswith('route')]

  hubs = gpd.read_file(hubs_shp_file)
  system = gpd.read_file(system_shp_file)

  X = pd.DataFrame()

  for i, route in enumerate(routes):

    X = X.append(gpx2df(route, hubs, system))
    print("==================")
    print("==================")
    print("==================")
    print("Cleaning route: ", i)
    print("==================")
    print("==================")
    print("==================")

  return X


def categorize_variables(dataframe):
  """
  """

  # Categorize start times.
  start_time_bins = [0, 5, 11, 16, 21, 24]
  start_times_labels = ['00:00-5:00', '5:00-11:00', '11:00-16:00',
                        '16:00-21:00', '21:00-00:00']
  dataframe['start_times_bins'] = pd.cut(dataframe['start_time'],
                                         start_time_bins,
                                         right=False,
                                         labels=start_times_labels,
                                         include_lowest=True)

  # Categorize Day of the weekday
  weekend_or_weekday_bins = [0, 5, 7]
  weekend_or_weekday_labels = ['weekday', 'weekend']
  dataframe['weekend_or_weekday_bins'] = pd.cut(dataframe['day_of_week'],
                                                weekend_or_weekday_bins,
                                                right=False,
                                                labels=weekend_or_weekday_labels,
                                                include_lowest=True)

  # Categorize ride time
  ride_time_bins = [0, 16, 31, 61, 10000]
  ride_time_labels = ['0-5m', '5-10m', '10-20m', '20m+']
  dataframe['ride_time_bins'] = pd.cut(dataframe['ride_time'],
                                       ride_time_bins,
                                       right=False,
                                       labels=ride_time_labels,
                                       include_lowest=True)

  # Categorize Heading
  dataframe.loc[dataframe.heading < 58, 'heading'] += 360
  heading_bins = [0, 148, 238, 328, 361+58]
  heading_labels = ['East', 'South', 'West', 'North']
  dataframe['heading_bins'] = pd.cut(dataframe['heading'],
                                     heading_bins,
                                     right=False,
                                     labels=heading_labels,
                                     include_lowest=True)

  # Return DF.
  return datafram