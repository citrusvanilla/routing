##
## documentation
##
"""
DOCSTRING
"""
from __future__ import division
import math
import os
from shapely.geometry import Polygon

import xml.etree.ElementTree as ET

import geopandas as gpd


# ================================

directory = os.path.join(os.getcwd(), "routes")

# ================================

def _deg2meters(degrees):
  """
  Helper function to convert degrees to meters.

  Args:
    degrees: float

  Returns:
    meters: float
  """

  meters = degrees * 111.325 * 1000

  return meters


def _meters2deg(meters):
  """
  Helper function to convert degrees to meters.

  Args:
    meters: float

  Returns:
    degrees: float
  """

  degrees = meters / 111.325 / 1000

  return degrees


def get_bounds(directory):
  """
  Returns bounds of a mesh.

  Args:
    directory: directory of GPX routes

  Returns: 
    max_lat: maximum lattitude as float in WGS84 coordinates
    max_lon: maximum longitude as float in WGS84 coordinates
    min_lat: minimum lattitude as float in WGS84 coordinates
    min_lon: minimum longitude as float in WGS84 coordinates
  """
  
  # Get a list of routes.
  routes = [i for i in os.listdir(directory) if i.startswith('route')]

  # Init bounds to none.
  max_lat, max_lon, min_lat, min_lon = None, None, None, None

  # Loop through all routes in the directory.
  for route in routes:

    # XML parser
    tree = ET.parse(os.path.join(directory, route))

    # Store the root of the XML tree.
    root = tree.getroot()

    # Get min/min attrs.
    cur_max_lat = float(root[1].get('maxlat'))
    cur_max_lon = float(root[1].get('maxlon'))
    cur_min_lat = float(root[1].get('minlat'))
    cur_min_lon = float(root[1].get('minlon'))

    # Update max lat, max lon, min lat, and min lon
    if max_lat is None: max_lat = cur_max_lat
    elif cur_max_lat > max_lat: max_lat = cur_max_lat

    if max_lon is None: max_lon = cur_max_lon
    elif cur_max_lon > max_lon: max_lon = cur_max_lon

    if min_lat is None: min_lat = cur_min_lat
    elif 40.657381 < cur_min_lat < min_lat: min_lat = cur_min_lat

    if min_lon is None: min_lon = cur_min_lon
    elif -74.140164 < cur_min_lon < min_lon: min_lon = cur_min_lon

  # Buffer bounds by 0.001 degrees (~100 meters).
  max_lat += 0.001
  max_lon += 0.001
  min_lat -= 0.001
  min_lon -= 0.001

  # Return vals.
  return max_lat, max_lon, min_lat, min_lon 


def build_mesh(directory, mesh_divisions):
  """
  Builds a mesh as a geopandas GeoDataFrame from a directory of GPX
  routes.  Mesh is guaranteed to cover all possible waypoints.

  Args:
    directory: directory containing GPX routes

  Returns:
    mesh: geopandas GeoDataFrame containing mesh regions as GeoSeries
  """

  # Get bounds of mesh.
  maxy, maxx, miny, minx = get_bounds(directory)

  # X and Y divisions counts.
  nx = mesh_divisions
  ny = mesh_divisions

  # X and Y divisions size.
  dx = abs(maxx - minx) / nx
  dy = abs(maxy - miny) / ny

  # Init mesh list and id counter.
  crs = {'init': 'epsg:4326'}
  mesh = gpd.GeoDataFrame(crs=crs)
  r_id = 0

  # For every "row" (lattitude) division:
  for i in range(ny):

    # For every "column" (longitude) division:
    for j in range(nx):

      # Init poly coors.
      vertices = []

      # Southwest corner coordinate:
      vertices.append([min(minx+dx*j,maxx),max(maxy-dy*i,miny)])

      # Southeast corner coordinate:
      vertices.append([min(minx+dx*(j+1),maxx),max(maxy-dy*i,miny)])

      # Northeast corner coordinate:
      vertices.append([min(minx+dx*(j+1),maxx),max(maxy-dy*(i+1),miny)])

      # Northwest corner coordinate:
      vertices.append([min(minx+dx*j,maxx),max(maxy-dy*(i+1),miny)])

      # Close loop, Southwest corner coordinate:
      vertices.append([min(minx+dx*j,maxx),max(maxy-dy*i,miny)])

      # Turn into a shapely Polygon
      r_poly = Polygon(vertices)

      # Init GeoSeries with Polygon
      r_series = gpd.GeoSeries(r_poly)
      r_series.name = r_id

      # Append Series to Mesh GeoDataFrame
      mesh = mesh.append(r_series)

      # Increase id.
      r_id += 1

  # Set gemotry.
  mesh = mesh.rename(columns={0: 'geometry'}).set_geometry('geometry')

  # Rotate the mesh.
  pass

  # Return the GeoDataFrame
  return mesh

