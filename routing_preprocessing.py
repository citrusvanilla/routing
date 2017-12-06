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
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


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
  start_times_labels = ['00:00-5:00', '5:00-11:00', '11:00-16:00',
                        '16:00-21:00', '21:00-00:00']
  dataframe['start_times_bins'] = pd.cut(dataframe['start_time'],
                                         start_time_bins,
                                         right=False,
                                         labels=start_times_labels,
                                         include_lowest=True)

  # Categorize weekend or weekday.
  weekend_or_weekday_bins = [0, 5, 7]
  weekend_or_weekday_labels = ['weekday', 'weekend']
  dataframe['weekend_or_weekday_bins'] = pd.cut(dataframe['day_of_week'],
                                                weekend_or_weekday_bins,
                                                right=False,
                                                labels=weekend_or_weekday_labels,
                                                include_lowest=True)

  # Categorize ride time.
  ride_time_bins = [0, 16, 31, 61, 10000]
  ride_time_labels = ['0-5m', '5-10m', '10-20m', '20m+']
  dataframe['ride_time_bins'] = pd.cut(dataframe['ride_time'],
                                       ride_time_bins,
                                       right=False,
                                       labels=ride_time_labels,
                                       include_lowest=True)

  # Categorize heading.
  dataframe.loc[dataframe.heading < 58, 'heading'] += 360
  heading_bins = [0, 148, 238, 328, 361+58]
  heading_labels = ['East', 'South', 'West', 'North']
  dataframe['heading_bins'] = pd.cut(dataframe['heading'],
                                     heading_bins,
                                     right=False,
                                     labels=heading_labels,
                                     include_lowest=True)

  # Return DF.
  return dataframe


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

    col_data = pd.get_dummies(col_data, prefix=col)
    out_df = out_df.join(col_data)

  return out_df


def trim_and_split_data(data, test_percentage, unlock_ooh=False, lock_ooh=False):
  '''Randomly shuffle the sample set.

  Args:
    data: pandas DF containing all ride breadcrumbs
    test_percentage: split percentage
    unlock_ooh: whether or not to include rides that started out-of-hub
    lock_ooh: whether or not to include rides that ended out-of-hub

  Returns:
    X_train:
    X_test:
    y_train:
    y_test:
  '''

  # Trim data
  if unlock_ooh is False:
    data = data[data.start_region != -1]

  if lock_ooh is False:
    data = data[data.end_region != -1]

  # Get rid of unwanted attributes
  data = data[[i for i in data.columns if i in ['current_region',
                                                'business district',
                                                'previous_region',
                                                'start_region',
                                                'end_region',
                                                'start_times_bins',
                                                'weekend_or_weekday_bins',
                                                'ride_time_bins',
                                                'heading_bins']]]

  # Split data into attributes and labels.
  X, y = data[[i for i in data.columns if i != 'end_region']], data['end_region']

  # Binarize X.
  X = binarize_features(X)

  # Binarize Y.
  lb = preprocessing.LabelBinarizer()
  lb.fit(y)
  y = lb.transform(y)

  # Split into test/train.
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=test_percentage,
                                                      random_state=0)

  return X_train, y_train, X_test, y_test


## ====================================================================


def preprocess(dataframe):
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

  # Handcraft categories from continuous attributes.
  dataframe = categorize_variables(dataframe)

  # Split data into test/train.
  X_train, y_train, X_test, y_test = trim_and_split_data(dataframe, 0.1)

  # Return
  return X_train, y_train, X_test, y_test

