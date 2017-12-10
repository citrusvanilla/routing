##
##  Predictive Routing
##  routing_metrics.py
##
##  Created by Justin Fung on 12/10/17.
##  Copyright 2017 Justin Fung. All rights reserved.
##
## ====================================================================
# pylint: disable=bad-indentation,bad-continuation,multiple-statements
# pylint: disable=invalid-name

"""
metrics.

Usage:
  Please see the README for how to compile the program and run the
  model.
"""

from __future__ import print_function
from __future__ import division

import pandas as pd
import re
import random


## ====================================================================


def break_ties(route_dataframe):
  """
  Break ties from classifier using a heuristic.  The heuristic is to choose
  the prediction from a tie if the prediction was our last breadcrumb's
  prediction.  That is, we are biasing the KNN prediction towards not
  making changes to destination predictions.

  Args:
    route_dataframe: pandas dataframe containing route breadcrumbs and
                     predictions.

  Returns:
    VOID
  """

  # Get a view of the predicted destination percentages only.
  percentage_view = route_dataframe[[col for col in route_dataframe
                                     if col.startswith('dest_')]]
  
  # Get the value of most-confident destination prediction.
  max_per_row = percentage_view.max(axis=1)

  # Loop through all the breadcrumbs in the route.
  for i, row in percentage_view.iterrows():
    
    # Init empty list to hold any prediction ties.
    max_destinations = []
    
    # Iterate across the prediction percentages.
    for dest in row.index:
      
      # If the prediction matches the highest...
      if max_per_row.loc[i]==row[dest]:
        
        # through it in the list.
        max_destinations.append(int(re.search("[0-9]+", dest).group(0)))
    
    # No precedent- choose randomly.
    if (len(max_destinations) > 1) and (i==0):
      route_dataframe.set_value(i, 'predicted_dest', random.choice(max_destinations))

    # If there are ties and one of the destinations in the tie was the
    # previous breadcrumbs prediction, make it the current prediction as well.
    if len(max_destinations) > 1 and \
       i > 0 and \
       route_dataframe.get_value(i-1, 'predicted_dest') in max_destinations:
      
      route_dataframe.set_value(
          i, 'predicted_dest', route_dataframe.get_value(i-1, 'predicted_dest'))
    else: # Random choice.
      route_dataframe.set_value(i, 'predicted_dest', random.choice(max_destinations))

  return


def get_inroute_accuracy(route_dataframe):
  """
  For any business district a rider enters, including his final
  destination, all prior predictions of that business district are
  considered correct predictions, i.e. if a rider makes a trip through
  both districts A and B, all predictions of A up to the last time the
  rider is in A, and all predictions of B up to the last time the rider
  is in B are considered accurate.

  Args:
    route_dataframe: pandas dataframe containing predicted breadcrumbs

  Returns:

  """

  # Get a business district view, and predictions as pd Series'.
  biz_dist_view = route_dataframe.current_bizdist
  predicted_dest_view = route_dataframe.predicted_dest

  # Init a dictionary to hold districts and last appearances.
  destination_dic = {}

  # Iterate through the view.
  for i, dest in biz_dist_view.iteritems():

    # Capture the district and the last time the rider was in district.
    if dest != -1:
      destination_dic[dest] = i

  # Init a new list, pd Series to hold inroute_accuracy.
  accuracy_scores = []

  # Iterate through predictions.
  for i, dest in predicted_dest_view.iteritems():

    # If breadcrumb prediction is in the actual route,
    # and occurs prior to arrival, add 1 to accuracy vector.
    if (dest in destination_dic) and (i <= destination_dic[dest]):
      accuracy_scores.append(1)
    else:
      accuracy_scores.append(0)

  # Init pd series to hold the scores.
  inroute_accuracy = pd.Series(accuracy_scores, name="inroute_accuracy")

  # Append to route dataframe.
  route_dataframe['inroute_accuracy'] = inroute_accuracy

  # Return an accuracy score.
  accuracy = round(inroute_accuracy.sum()/len(inroute_accuracy), 2)
  print("In-route accuracy for route ", str(int(route_dataframe.route_id[0])),
        ": ", accuracy*100, "%.", sep="")

  return accuracy


def get_final_dest_accuracy(route_dataframe):
  """
  final dest acc

  Args:
    route_dataframe: pandas dataframe containing predicted breadcrumbs

  Returns:

  """
  # Get a business district view, and predictions as pd Series'.
  final_destination = route_dataframe.current_bizdist[len(route_dataframe)-1]
  predicted_dest_view = route_dataframe.predicted_dest

  # Init a new list, pd Series to hold inroute_accuracy.
  accuracy_scores = []

  # Iterate through predictions.
  for i, destination in predicted_dest_view.iteritems():

    # If breadcrumb prediction is in the actual route,
    # and occurs prior to arrival, add 1 to accuracy vector.
    if destination == final_destination:
      accuracy_scores.append(1)
    else:
      accuracy_scores.append(0)

  # Init pd series to hold the scores.
  final_destination_accuracy = pd.Series(accuracy_scores,
                                         name="final_destination_accuracy")

  # Append to route dataframe.
  route_dataframe["final_destination_accuracy"] = final_destination_accuracy

  # Return an accuracy score.
  accuracy = round(final_destination_accuracy.sum()/
                   len(final_destination_accuracy), 2)
  print("Final destination accuracy for route ",
        str(int(route_dataframe.route_id[0])),
        ": ", accuracy*100, "%.", sep="")

  return


## ====================================================================


def get_metrics(route_dataframe, full_dataset):
  """
  blah blah blah
  """

  # Break ties.
  break_ties(route_dataframe)

  # Set in-route accuracy Series and get accuracy score.
  inroute_acc = get_inroute_accuracy(route_dataframe)

  # Set final destination accuracy Series and get accuracy score.
  final_acc = get_final_dest_accuracy(route_dataframe)

  # Return.
  return route_dataframe, inroute_acc, final_acc

