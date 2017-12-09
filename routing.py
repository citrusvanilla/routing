##
##  Predictive Routing
##  routing.py
##
##  Created by Justin Fung on 12/10/17.
##  Copyright 2017 Justin Fung. All rights reserved.
##
## ====================================================================
# pylint: disable=bad-indentation,bad-continuation,multiple-statements
# pylint: disable=invalid-name

"""
Command line module for predicting final destinations of a GPX route,
based on a KNN model/database.

Usage:
  Please see the README for how to compile the program and run the
  model.
"""

from __future__ import print_function

import os
import re
import time
import sys
import getopt

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import routing_io
import routing_preprocessing
import routing_output
import routing_metrics


## ====================================================================


routes_directory = os.path.join(os.getcwd(), "routes")
pickled_data_path = os.path.join(os.getcwd(), "data", "data.pkl")
prediction_directory = os.path.join(os.getcwd(), "output")


## ====================================================================


def get_labels_from_probabilities(array_of_probabilities):
  """
  """

  # Init empty list for labels.
  labels = []

  for i in range(len(array_of_probabilities[0])):

    max_score = 0
    max_idx = "tie"

    for j in range(len(array_of_probabilities)):

      cur_score = array_of_probabilities[j][i][1]

      if cur_score > max_score:
        max_score = cur_score
        max_idx = j
      elif cur_score == max_score:
        max_idx = "tie"

    labels.append(max_idx)

  return labels


def train(X_train, y_train):

  """
  Train classifier.
  """

  # Initialize and return a classifier with the train/test data.
  print("Training a classifier...")
  start_time = time.time()

  classifier = KNeighborsClassifier(n_neighbors=7)
  classifier.fit(X_train[[i for i in X_train.columns
                          if i not in ["idx", "route_id"]]],
                 y_train[[i for i in y_train.columns
                          if i not in ["idx", "route_id"]]])

  print("Classifier trained in ", time.time()-start_time, " seconds.", sep="")

  return classifier


def predict(route_id, data):
  """
  Predict route destination.
  """

  # Preprocess, split, and return the data for learning.
  X_train, y_train, X_test, y_test = routing_preprocessing.\
                                     preprocess(route_id, data)

  # Initialize a classifier.
  classifier = train(X_train, y_train)

  # Make the prediction.
  y_pred = classifier.predict(X_test[[i for i in X_test.columns
                                      if i not in ["idx", "route_id"]]])

  y_pred_prob = classifier.predict_proba(
                              X_test[[i for i in X_test.columns
                                      if i not in ["idx", "route_id"]]])

  # Hack! Seems to be an issue with skikit's implementation of KNN for
  # multi-label classification!
  labels = get_labels_from_probabilities(y_pred_prob)

  # Join labels with raw data.
  outdata = data.loc[data.idx.isin(X_test.idx)]
  outdata = outdata.copy(deep=True)
  outdata['predicted_dest'] = pd.DataFrame(labels).values

  # Join label probabilities with raw data.
  prob_dic = {}

  # Loop through all destinations.
  for i in range(len(y_pred_prob)):

    # Set the key and the pair.
    key = "dest_" + str(i) + "_perc"
    value = y_pred_prob[i][:, 1]

    # Assign to the dictionary.
    prob_dic[key] = value

  # Convert dictionary to dataframe.
  probabilities = pd.DataFrame(prob_dic)
  probabilities.reset_index(drop=True, inplace=True)

  # Concatenate with results.
  outdata.reset_index(drop=True, inplace=True)
  result = pd.concat([outdata, probabilities], axis=1)

  return result


## ====================================================================


def main(argv):
  """
  Main.
  """
  # The command line should have one argument- the path to a gpx file.
  inputfile = ''
  try:
    opts, args = getopt.getopt(argv, "i:")
  except getopt.GetoptError:
    print("usage: routing.py -i <path to gpx file>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == ("-i"):
      inputfile = arg

  # Check for routes directory.
  if not os.path.exists(routes_directory):
    print("No Routes directory!  Contact Author for access.")
    return -1

  # Check for binarized pandas dataframe data.
  if not os.path.exists(pickled_data_path):
    print("Building Pandas Dataframe from Routes directory. "
          "This make take awhile.", sep="")
    routing_io.build_X(routes_directory)

  # Load data and predict.
  data = pd.read_pickle(pickled_data_path)

  # Extract route_id from command line arg.
  route_file = os.path.basename(inputfile)
  route_id = int(re.search("[0-9]+", route_file.split(".")[0]).group(0))

  # Predict destination for the route.
  df_route = predict(route_id, data)

  # Get metrics.  Print out.
  df_route, inroute_acc, final_acc = routing_metrics.get_metrics(df_route, data)

  # Write out GPX and log.
  routing_output.output(df_route, prediction_directory)


if __name__ == "__main__":
  main(sys.argv[1:])

