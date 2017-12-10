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

from __future__ import print_function
from __future__ import division

import csv
import os
import re
import numpy as np

from datetime import datetime
import xml.etree.ElementTree as ET
import fiona

import routing_preprocessing


## ====================================================================


hubs = {-1: "None",
        0: "Monroe St & 8th St",
        1: "Washington St & 14th St",
        2: "14th St Pier",
        3: "Church Square Park",
        4: "Washington St & 3rd St",
        5: "Hudson St & Hudson",
        6: "PATH & Pier A"}

business_districts = {-1: "None",
                      0: "The Ferry",
                      1: "Downtown",
                      2: "Uptown & The Pier",
                      3: "The Light Rail"}


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
  # Init a root.
  root = ET.Element(tag="gpx",
                    attrib={"creator": "Social Bicycles - Justin Fung"})

  # Init a tree with the root.
  tree = ET.ElementTree(element=root)

  # Init top-level GPX-required elements.
  # GPX "time" element.
  gpx_time = ET.Element(tag="time")
  gpx_time.text = dataframe.datetime_raw[0]

  # GPX "bounds" element
  gpx_bounds = ET.Element(tag="bounds",
                          attrib={"minlat": dataframe.minlat[0],
                                  "minlon": dataframe.minlon[0],
                                  "maxlat": dataframe.maxlat[0],
                                  "maxlon": dataframe.maxlon[0]})
  # GPD trk element.
  gpx_trk = ET.Element(tag="trk")

  # Init top-level handcrafted elements.
  # HC "route id" element.
  hc_route_id = ET.Element(tag="route_id")
  hc_route_id.text = str(int(dataframe.route_id[0]))

  # HC "start_hour" element.
  hc_time = ET.Element(
             tag="hc_time",
             attrib={"weekend_or_weekday": dataframe.weekend_or_weekday_bins[0],
                     "starting_hour": dataframe.start_times_bins[0]})

  # HC "locations" element.
  hc_locations = ET.Element(
                  tag="hc_locations",
                  attrib={"start_hub": str(int(dataframe.start_region[0])),
                          "end_hub": str(int(dataframe.end_region[0])),
                          "end_biz_dist": str(int(
                                          dataframe.end_business_district[0]))})

  # Add top-level elements as children of the root.
  root.extend([gpx_time, gpx_bounds, hc_route_id, hc_time, hc_locations,
               gpx_trk])

  # Build out the "trk" element with DataFrame breakcrumbs.
  gpx_trkseg = ET.Element(tag="trkseg")
  gpx_trk.append(gpx_trkseg)

  # Loop through all the breadcrumbs to build the track segment points.
  for index, breadcrumb in dataframe.iterrows():

    # Init a trkpt element.
    gpx_trkpt = ET.Element(tag="trkpt", attrib={"lat": str(breadcrumb.lat),
                                                "lon": str(breadcrumb.lon)})
    # "Ride type" subelement.
    gpx_type = ET.Element(tag="type")
    gpx_type.text = breadcrumb.ride_type

    # "Current Location" subelement.
    hc_curr_loc = ET.Element(tag="current_loc")
    if breadcrumb.current_region is None:
      hc_curr_loc.text = "None"
    else:
      hc_curr_loc.text = str(int(breadcrumb.current_region))

    # "Current Business District" subelement
    hc_bizdist = ET.Element(tag="current_bizdist")
    hc_bizdist.text = str(int(breadcrumb.current_bizdist))

    # "Previous Location" subelement.
    hc_prev_loc = ET.Element(tag="previous_loc")
    if breadcrumb.previous_region is None:
      hc_prev_loc.text = "None"
    else:
      hc_prev_loc.text = str(int(breadcrumb.previous_region))

    # "Heading" subelement.
    hc_heading = ET.Element(tag="heading")
    if isinstance(breadcrumb.heading_bins, float) and \
       np.isnan(breadcrumb.heading_bins):
      hc_heading.text = "None"
    else:
      hc_heading.text = breadcrumb.heading_bins

    # "Ride Time" subelement.
    hc_ride_time = ET.Element(tag="ride_time")
    hc_ride_time.text = breadcrumb.ride_time_bins

    # "Predicted Destination" subelement.
    hc_predicted_dest = ET.Element(tag="predicted_dest")
    if isinstance(breadcrumb.predicted_dest, str):
      hc_predicted_dest.text = "Tie"
    else:
      hc_predicted_dest.text = str(int(breadcrumb.predicted_dest))

    # "Potential Destination" percentages subelement.
    destination_percentages_dic = {}

    for idx, idxdata in breadcrumb.iteritems():
      if idx.startswith("dest"):
        key = str(int(re.search("[0-9]+", idx).group(0)))
        value = str(idxdata)
        destination_percentages_dic[key] = value

    hc_dest_perc = ET.Element(tag="destination_per",
                              attrib=destination_percentages_dic)

    # Add subelements to trkpt element.
    gpx_trkpt.extend([gpx_type, hc_curr_loc, hc_prev_loc, hc_heading,
                      hc_bizdist, hc_ride_time, hc_predicted_dest,
                      hc_dest_perc])

    # Append Breadcrumb ("trkpt") element to trkseg element.
    gpx_trkseg.append(gpx_trkpt)

  # Pretty-print tree and return.
  indent(root)

  return tree


def write_gpx(elementtree, directory):
  """
  Writes out a xml.etree.elementtree Element Tree to GPX(XML) file.

  Args:
    elementree: an xml.etree.elementtree Element Tree object.

  Returns:
    VOID
  """

  # Get route id for filename.
  route_id = elementtree.getroot().find("route_id").text
  gpx_path = os.path.join(directory, "route_" + route_id + ".gpx")

  # Try/except a IO method from xml.etree.
  try:
    elementtree.write(gpx_path,
                      encoding="UTF-8",
                      xml_declaration=True,
                      default_namespace=None,
                      method="xml")
  except:
    print("Could not write out GPX file.")

  return


def write_csv(dataframe, directory):
  """
  """
  # Convert column vales to names.
  dataframe.end_business_district = dataframe.end_business_district.astype(int)
  dataframe.current_bizdist = dataframe.current_bizdist.astype(int)
  dataframe.end_region = dataframe.end_region.astype(int)
  dataframe.start_region = dataframe.start_region.astype(int)
  dataframe.predicted_dest = dataframe.predicted_dest.astype(int)

  dataframe = dataframe.replace({"end_business_district": business_districts})
  dataframe = dataframe.replace({"current_bizdist": business_districts})
  dataframe = dataframe.replace({"end_region": hubs})
  dataframe = dataframe.replace({"start_region": hubs})
  dataframe = dataframe.replace({"predicted_dest": business_districts})

  dataframe.displacement = dataframe.displacement .round(1)

  # Rename the columns.
  out_cols = ["lat", "lon", "current_region", "heading_bins",
              "displacement", "current_bizdist", "predicted_dest",
              "dest_0_perc", "dest_1_perc", "dest_2_perc", "dest_3_perc"]
  confidence_cols = {}
  for i in out_cols:
    if i.startswith("dest_"):
      confidence_cols[i] = "Confidence in " + \
                    business_districts[int(re.search("[0-9]+", i).group(0))]
  out_df = dataframe[[i for i in out_cols]]
  out_df = out_df.rename(index=str,
                         columns={"lat": "Lattitude",
                                 "lon": "Longitude",
                                 "current_region": "Current System Area",
                                 "heading_bins": "Current Heading",
                                 "displacement": "Movement Since Last Update (meters)",
                                 "current_bizdist": "Current Destination",
                                 "predicted_dest": "Predicted Final Destination"})
  out_df = out_df.rename(index=str, columns=confidence_cols)

  # Get some metrics.
  inroute_accuracy = round(dataframe.inroute_accuracy.sum()/
                           len(dataframe.inroute_accuracy), 2)


  finaldest_accuracy = round(dataframe.final_destination_accuracy.sum()/
                             len(dataframe.final_destination_accuracy), 2)

  # Write out.
  out_path = os.path.join(directory, "route_log.csv")
  with open(out_path, "wb") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["Time of Report", str(datetime.now())])
    writer.writerow(["Route ID", dataframe.route_id[0]])
    writer.writerow(["Time of Ride Start", dataframe.datetime_raw[0]])
    writer.writerow(["Weekday or Weekend", dataframe.weekend_or_weekday_bins[0]])
    writer.writerow(["Time of Day", dataframe.start_times_bins[0]])
    writer.writerow(["Total Ride Time",
                     str(round(len(dataframe)*30/60,1)) + " minutes"])
    writer.writerow(["Start hub", dataframe.start_region[0]])
    writer.writerow(["End hub", dataframe.end_region[0]])
    writer.writerow(["Final destination", dataframe.end_business_district[0]])
    writer.writerow(["In-route accuracy", inroute_accuracy])
    writer.writerow(["Final destination accuracy", finaldest_accuracy])
    writer.writerow([])
    writer.writerow([])

  with open(out_path, "a") as f:
    out_df.to_csv(f, header=True)

  f.close()



## ====================================================================


def output(dataframe, output_directory):
  """
  Writes out a df.
  """
  # Process columns for output.
  dataframe = routing_preprocessing.categorize_variables(dataframe)

  # Build GPX Tree.
  tree = build_gpx_tree(dataframe)

  # Write the tree to GPX file.
  write_gpx(tree, output_directory)

  # Write a CSV.
  write_csv(dataframe, output_directory)


