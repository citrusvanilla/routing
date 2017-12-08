##
##  Predictive Routing
##  routing_train.py
##
##  Created by Justin Fung on 12/10/17.
##  Copyright 2017 Justin Fung. All rights reserved.
##
## ====================================================================
# pylint: disable=bad-indentation,bad-continuation,multiple-statements
# pylint: disable=invalid-name

"""


Usage:
  Please see the README for how to compile the program and run the
  model.
"""

# Load libraries
from datetime import datetime
import math
import os
import re

import xml.etree.ElementTree as ET
import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point


## ====================================================================


def indent(elem, level=0):
  """
  Helper function for pretty-printing GPX file.
  """
  i = ("\n" + level*"  ")
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = (i + "  ")
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
    for elem in elem:
      indent(elem, level+1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = i


def build_gpx_tree(dataframe):
  """
  Build a Python xml.etree.ElementTree ElementTree object from a pandas
  dataframe, representing a full GPX ride.

  Args:
    dataframe: pandas dataframe

  Returns:
    tree: xml.etree.ElementTree ElementTree object
  """
  # Get some attrs.
  raw_time = dataframe.datetime_raw[0]
  bounds = {"minlat": dataframe.minlat[0],
            "minlon": dataframe.minlon[0],
            "maxlat": dataframe.maxlat[0],
            "maxlon": dataframe.maxlon[0]}
  datetime = {"weekend_or_weekday": dataframe.weekend_or_weekday_bins[0],
              "starting_hour": dataframe.start_times_bins[0]}
  locations = {"start_hub": str(int(dataframe.start_region[0])),
                "end_hub": str(int(dataframe.end_region[0])),
                "end_biz_dist": str(int(dataframe.end_business_district[0]))}

  # Init a root.
  root = ET.Element(tag="gpx",
                    attrib={"creator": "Social Bicycles - Justin Fung"})

  # Init a tree with the root.
  tree = ET.ElementTree(element=root)

  # Init top-level GPX-required elements.
  # GPX "time" element.
  gpx_time = ET.Element(tag="time")
  gpx_time.text = raw_time

  # GPX "bounds" element
  gpx_bounds = ET.Element(tag="bounds", attrib=bounds)

  # GPD trk element.
  gpx_trk = ET.Element(tag="trk")

  # Init top-level handcrafted elements.
  # HC "start_hour" element.
  hc_time = ET.Element(tag="hc_time", attrib=datetime)

  # HC "locations" element.
  hc_locations = ET.Element(tag="hc_locations", attrib=locations)

  # Add top-level elements as children of the root.
  root.extend([gpx_time, gpx_bounds, hc_time, hc_locations, gpx_trk])

  # Build out the "trk" element with DataFrame breakcrumbs.
  gpx_trkseg = ET.Element(tag="trkseg")
  gpx_trk.append(gpx_trkseg)

  # Loop through all the breadcrumbs to build the track segment points.
  for index, breadcrumb in dataframe.iterrows():

    # Init a trkpt element.
    lat_lon = {"lat": str(breadcrumb.lat),
               "lon": str(breadcrumb.lon)}
    gpx_trkpt = ET.Element(tag="trkpt", attrib=lat_lon)

    # Ride Type subelement.
    gpx_type = ET.Element(tag="type")
    gpx_type.text = breadcrumb.ride_type

    # Current Location subelement.
    hc_curr_loc = ET.Element(tag="current_loc")
    if breadcrumb.current_region is None:
      hc_curr_loc.text = "None"
    else:
      hc_curr_loc.text = str(int(breadcrumb.current_region))

    # Previous Location subelement.
    hc_prev_loc = ET.Element(tag="previous_loc")
    if breadcrumb.previous_region is None:
      hc_prev_loc.text = "None"
    else:
      hc_prev_loc.text = str(int(breadcrumb.previous_region))

    # Heading subelement.
    hc_heading = ET.Element(tag="heading")
    if isinstance(breadcrumb.heading_bins, float) and \
       np.isnan(breadcrumb.heading_bins):
      hc_heading.text  = "None"
    else:
      hc_heading.text = breadcrumb.heading_bins

    # Ride Time subelement.
    hc_ride_time = ET.Element(tag="ride_time")
    hc_ride_time.text = breadcrumb.ride_time_bins

    # Add subelements to trkpt element.
    gpx_trkpt.extend([gpx_type, hc_curr_loc, hc_prev_loc, hc_heading,
                      hc_ride_time])

    # Append Breadcrumb ("trkpt") element to trkseg element.
    gpx_trkseg.append(gpx_trkpt)

  # Return tree.
  indent(root)

  return tree


def write_gpx(elementtree):
  """
  Writes out a xml.etree.elementtree Element Tree to GPX(XML) file.

  Args:
    elementree: an xml.etree.elementtree Element Tree object.

  Returns:
    VOID
  """

  # Try/except a IO method from xml.etree.
  try:
    elementtree.write("test.gpx",
                      encoding="UTF-8",
                      xml_declaration=True,
                      default_namespace=None,
                      method="xml")
  except:
    print("Could not write out GPX file.")

  return


def output(dataframe, output_directory):
  """
  Writes out a df.
  """
  # Build GPX Tree.
  tree = build_gpx_tree(dataframe)

  

  # Clean it up.




