##
##  Predictive Routing
##  routing_preprocessing.py
##
##  Created by Justin Fung on 12/10/17.
##  Copyright 2017 Justin Fung. All rights reserved.
##
## ====================================================================
# pylint: disable=bad-indentation,bad-continuation,multiple-statements
# pylint: disable=invalid-name

"""
Module for preprocessing a pandas dataframe containing extracted GPX
routes for a machine learning classifier.

Usage:
  Please see the README for how to compile the program and run the
  model.
"""

# Load libraries
import pandas as pd


## ====================================================================


def categorize_variables(dataframe):
  """
  Turns continuous attributes for the following into categorical attrs.

  Args:
    dataframe: pandas DF without categorical attrs

  Returns:
    dataframe: pandas DF with categorical attrs
  """

  # Categorize start times.
  start_time_bins = [0, 5, 11, 16, 21, 24]
  start_times_labels = ["00:00-5:00", "5:00-11:00", "11:00-16:00",
                        "16:00-21:00", "21:00-00:00"]
  dataframe["start_times_bins"] = pd.cut(dataframe["start_time"],
                                         start_time_bins,
                                         right=False,
                                         labels=start_times_labels,
                                         include_lowest=True)

  # Categorize weekend or weekday.
  weekend_or_weekday_bins = [0, 5, 7]
  weekend_or_weekday_labels = ["weekday", "weekend"]
  dataframe["weekend_or_weekday_bins"] = pd.cut(dataframe["day_of_week"],
                                                weekend_or_weekday_bins,
                                                right=False,
                                                labels=weekend_or_weekday_labels,
                                                include_lowest=True)

  # Categorize ride time.
  ride_time_bins = [0, 16, 31, 61, 10000]
  ride_time_labels = ["0-5m", "5-10m", "10-20m", "20m+"]
  dataframe["ride_time_bins"] = pd.cut(dataframe["ride_time"],
                                       ride_time_bins,
                                       right=False,
                                       labels=ride_time_labels,
                                       include_lowest=True)

  # Categorize heading.
  dataframe.loc[dataframe.heading < 58, "heading"] += 360
  heading_bins = [0, 148, 238, 328, 361+58]
  heading_labels = ["East", "South", "West", "North"]
  dataframe["heading_bins"] = pd.cut(dataframe["heading"],
                                     heading_bins,
                                     right=False,
                                     labels=heading_labels,
                                     include_lowest=True)

  # Return DF.
  return dataframe.drop(["start_time",
                         "day_of_week",
                         "ride_time",
                         "heading"], axis=1)


def binarize_features(dataframe):
  """
  Turns categorical attributes into bernoulli dummy variables.

  Args:
    dataframe: pandas DF holding all ride breadcrumbs

  Returns:
    out_df: expanded pandas DataFrame
  """

  # Init dataframe for output.
  out_df = pd.DataFrame(index=dataframe.index)

  # Iterate through all columns and expand.
  for col, col_data in dataframe.iteritems():

    if col not in  ["idx", "route_id"]:
      col_data = pd.get_dummies(col_data, prefix=col)
    out_df = out_df.join(col_data)

  return out_df


def remove_excess(dataframe):
  """
  Remove breadcrumbs that do not contribute to learning.
  """

  # Subset breadcrumbs that are not starts of rides.
  bad_crumbs = dataframe[dataframe.route_id ==
                         dataframe.shift(periods=1).route_id]

  # Subset breadcrumbs in which rider did not move from current region.
  bad_crumbs = bad_crumbs[bad_crumbs.current_region ==
                          bad_crumbs.shift(periods=1).current_region]

  # Subset crumbs in which the GPS position did not appreciably change.
  bad_crumbs = bad_crumbs[bad_crumbs.displacement <= 30]

  # Drop all the bad crumbs from the Dataframe view.
  clean_dataframe = dataframe.drop(bad_crumbs.index)

  return clean_dataframe


def trim_and_split_data(data, holdout_route, biz_or_hub=0):
  """Randomly shuffle the sample set.

  Args:
    data: pandas DF containing all ride breadcrumbs
    test_percentage: split percentage
    unlock_ooh: whether or not to include rides that started out-of-hub
    lock_ooh: whether or not to include rides that ended out-of-hub

  Returns:
    X_train: routes for training
    X_test: routes for testing
    y_train: labels for training
    y_test: labels for testing
  """

  # Split into test/train by route id.
  all_routes = list(data.route_id.unique())
  test_routes = [holdout_route]
  train_routes = list(set(all_routes).difference(set(test_routes)))

  testing_data = data.loc[data.route_id.isin(test_routes)]
  training_data = data.loc[data.route_id.isin(train_routes)]

  # Clean out training data.
  training_data = remove_excess(training_data)

  # Combine views again for further processing.
  data_comb = data.loc[list(testing_data.index) + list(training_data.index)]

  # Trim data
  if biz_or_hub == 0:
    data_comb = data_comb[data_comb.end_business_district != -1]
    training_attrs = ["end_business_district", "current_region", "day_of_week",
                      "heading", "previous_region", "ride_time",
                      "start_region", "start_time", "idx", "route_id"]
    label = "end_business_district"
  elif biz_or_hub == 1:
    data_comb = data_comb[data_comb.end_region != -1]
    training_attrs = ["current_region", "day_of_week",
                      "end_region", "heading", "previous_region", "ride_time",
                      "start_region", "start_time", "idx", "route_id"]
    label = "end_region"

  # Keep only training attrs.
  data_comb = data_comb[[i for i in data_comb.columns if i in training_attrs]]

  # Categorize vars.
  data_comb = categorize_variables(data_comb)

  # Split data into X, y.
  X = data_comb[[i for i in data_comb.columns if i != label]]
  y = data_comb[[label, "route_id", "idx"]]

  # Binarize categorical data.
  X = binarize_features(X)
  y = binarize_features(y)

  # Split test/train.
  X_train = X.loc[X.route_id.isin(train_routes)]
  X_test = X.loc[X.route_id.isin(test_routes)]
  y_train = y.loc[y.route_id.isin(train_routes)]
  y_test = y.loc[y.route_id.isin(test_routes)]

  return X_train, y_train, X_test, y_test


## ====================================================================


def preprocess(route_id_to_predict, dataframe):
  """
  Preprocesses raw datframe of GPX breadcrumbs into cluster-friendly
  formatting.

  Args:
    dataframe: pandas dataframe with raw gpx breadcrumbs

  Returs:
    X_train: explanatory vars for training
    y_train: label for training
    X_test: explanatory vars for test
    y_test: label for testing
  """

  # Split data into test/train.
  X_train, y_train, X_test, y_test = trim_and_split_data(dataframe,
                                                         route_id_to_predict, 0)



  # Return
  return X_train, y_train, X_test, y_test

