##
##  Predictive Routing
##  routing_io.py
##
##  Created by Justin Fung on 12/10/17.
##  Copyright 2017 Justin Fung. All rights reserved.
##
## ====================================================================
# pylint: disable=bad-indentation,bad-continuation,multiple-statements
# pylint: disable=invalid-name

"""
Module for converting an external directory of GPX files to a pandas
dataframe for preprocessing and model training.

Usage:
  Please see the README for how to compile the program and run the model.
"""

from datetime import datetime
import math
import os
import re

import xml.etree.ElementTree as ET
import fiona
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point


## ====================================================================


routes_directory = os.path.join(os.getcwd(), "routes")

hubs_shp_file = os.path.join(os.getcwd(), "data", "hubs.shp")
system_shp_file = os.path.join(os.getcwd(), "data", "system.shp")
bizdist_shp_file = os.path.join(os.getcwd(), "data", "business_districts.shp")


## ====================================================================


def get_region(point, mesh, crs):
  """
  Spatial-joins a lat/lon coordinate in CRS 4326 to a polygon mesh of
  the same coordinate system.

  Args:
    point: (longitude, lattitude) as a tuple
    mesh: mesh as a GeoPandas DataFrame
    crs: coordinate system to use

  Returns:
    joined_region: region id if join, else -1
  """

  # Create a Shapely Point object.
  point = Point(point[0], point[1])

  # Convert to a GeoPandas DataFrame and defined the geometery.
  series = gpd.GeoSeries(point, crs=crs, name="breadcrumb")
  df = gpd.GeoDataFrame().append(series)
  df = df.rename(columns={0: 'geometry'}).set_geometry('geometry')

  # Spatial join the Point DF to the mesh DF.
  region = gpd.sjoin(df, mesh, how="inner", op='intersects')

  # Return the region if spatial join successful, else -1.
  if len(region) > 0:
    joined_region = region.index_right[0]
  else:
    joined_region = -1

  return joined_region


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.

    Args:
      pointA: start point as tuple
      pointB: end point as tuple

    Returns:
      compass_bearing: bearing as float between 0 and 360
    """

    # Calculate initial bearing.
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Normalize the initial bearing and return.
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees).

    Args:
      lon1: longitude in decimal degrees
      lat1: lattitude in decimal degrees
      lon2: longitude in decimal degrees
      lat2: lattitude in decimal degrees

    Returns:
      meters: haversine distance as float
    """

    # Convert decimal degrees to radians.
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Apply Haversine formula.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in km is 6371, convert to meters and return.
    meters = 6371000 * c

    return meters


def gpx2df(gpx_file, hubs, business_district, system):
  """
  Converts one gpx file to a pandas dataframe representing a gpx track.

  Args:
    gpx_file: path to a gpx file
    hubs: geopandas dataframe representing bike share hubs
    business_district: geopandas dataframe representing business
                       districts
    system: geopandas dataframe representing bike share system mesh

  Returns:
    ride_df: geodataframe holding all breadcrumbs and extracted attrs.
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

  # Get business district.
  business_district = get_region((float(track_segment[len(track_segment)-1].get('lon')),
                                  float(track_segment[len(track_segment)-1].get('lat'))),
                                  business_district, crs)

  # Init counters and pointers.
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
    if count > 0 and displacement > 30:
      heading = calculate_initial_compass_bearing(prev_pts, bc_pts)

    # Init Pandas Series.
    pseries = pd.Series(data=[gpx_file, route_id, start_time, weekday,
                              start_region, end_region, count,
                              previous_physical_region, current_region, heading,
                              displacement, business_district],
                        index=['gpx file', 'route id', 'start_time',
                               'day_of_week', 'start_region', 'end_region',
                               'ride_time', 'previous_region', 'current_region',
                               'heading', 'displacement', 'business district'])

    # Append Ride to Dataframe.
    ride_df = ride_df.append(pseries, ignore_index=True)

    # Update previous breadcrumb.
    prev_pts = bc_pts

    # Update pointer to last region.
    previous_time_region = current_region

  # Exit.
  return ride_df


def build_X(routes_dir):
  """
  Builds a master dataframe to hold geodataframes representing mulitple
  GPX routes in a route directory.

  Args:
    routes_dir: directory holding GPX routes

  Returns:
    X: master geodataframe
  """

  # Get file names.
  routes = [i for i in os.listdir(routes_dir) if i.startswith('route')]

  # Read in hub, system, business district shapefiles to GeoDataFrames.
  hubs = gpd.read_file(hubs_shp_file)
  system = gpd.read_file(system_shp_file)
  bizdist = gpd.read_file(bizdist_shp_file)

  # Init empty DF to hold extracts GPX routes.
  X = pd.DataFrame()

  # Loop through the routes and append.
  for i, route in enumerate(routes):

    X = X.append(gpx2df(route, hubs, bizdist, system))
    print("==================")
    print("==================")
    print("==================")
    print("Cleaned route: ", i)
    print("==================")
    print("==================")
    print("==================")

  # Reset the indexing and return.
  X = X.reset_index(drop=True)

  return X


## ====================================================================


def main():
  """
  Builds GeoDataFrame holding extracted GPX routes and returns.
  """

  return build_X(routes_directory)

