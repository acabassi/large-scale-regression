import os
import sys
import pandas
import functools
import numpy as np
import tensorflow as tf

### Define input format

CSV_COLUMNS = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime',
    'UniqueCarrier', 'FlightNum',  'ArrDelay', 'DepDelay', 'Origin', 'Dest',
    'Distance', 'Cancelled', 'Diverted']
DEFAULTS = [[""], [""], [""], [""], [0], [0], [""], [""], [0.], [0.],[""], [""],
    [0], [""],[""]]
LABEL_COLUMN = 'ArrDelay'

### Parse csv file(s)

# If you want to train your dataset only on one year's data
train_file = "2006.csv"
# If you want to use all the data
# train_file = "*.csv"

### Define parser

def parse_csv(value):
      tf.logging.info('Parsing {}'.format(data_file))
      columns = tf.decode_csv(value, record_defaults=DEFAULTS, select_cols = [0,
        1, 2, 3, 4, 6, 8, 9, 14, 15, 16, 17, 18, 19, 21], na_value="NA")
      features = dict(zip(CSV_COLUMNS, columns))
      labels = features.pop('ArrDelay')
      # Define the two classes for logistic regression
      # If the DepDelay is greater than 0, than the label is True (i.e. the flight
      # was delayed). Otherwise, it is False.
      classes = tf.greater(labels, 0)
      return features, classes

### Define input function

def input_fn(data_file, num_epochs, shuffle, batch_size, buffer_size=1000):
      # Create list of file names that match "glob" pattern
      # (i.e. data_file_*.csv)
      filenames_dataset = tf.data.Dataset.list_files(data_file)
      # Read lines from text files
      textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
      # Parse text lines as comma-separated values (CSV)
      dataset = textlines_dataset.map(parse_csv)
      if shuffle:
          dataset = dataset.shuffle(buffer_size=buffer_size)
      # We call repeat after shuffling, rather than before, to prevent separate
      # epochs from blending together.
      dataset = dataset.repeat(num_epochs)
      # Get a batch of data of size bathc_size
      dataset = dataset.batch(batch_size)
      return dataset

### Define wrapper for input function for training step

train_inpf = functools.partial(input_fn, train_file, num_epochs=1, shuffle=True,
    batch_size=100)

### Define columns

import tensorflow.feature_column as fc

year = fc.categorical_column_with_vocabulary_list('Year',
    ['1987', '1988', '1989', '1990', '1991', '1992', '1993',
    '1994', '1995', '1996', '1997', '1998', '1999', '2000',
    '2001', '2002', '2003', '2004', '2005', '2006', '2007',
    '2008'])
month = fc.categorical_column_with_vocabulary_list('Month',
    ['1','2','3','4','5','6','7','8','9','10','11','12'])
dayofmonth = fc.categorical_column_with_vocabulary_list('DayofMonth',
    ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15',
    '16','17','18','19','20', '21', '22', '23', '24', '25', '26', '27',
    '28', '29','30', '31'])
dayofweek = fc.categorical_column_with_vocabulary_list('DayOfWeek',
    ['1','2','3','4','5','6','7'])
deptime = fc.numeric_column('DepTime')
arrtime = fc.numeric_column('ArrTime')
uniquecarrier = fc.categorical_column_with_hash_bucket('UniqueCarrier',
    hash_bucket_size=1000)
flightnum = fc.categorical_column_with_hash_bucket('FlightNum',
    hash_bucket_size=10000)
arrdelay = fc.numeric_column('ArrDelay')
depdelay = fc.numeric_column('DepDelay')
origin = fc.categorical_column_with_hash_bucket('Origin',
    hash_bucket_size=1000)
dest = fc.categorical_column_with_hash_bucket('Dest',
    hash_bucket_size=1000)
distance = fc.numeric_column('Distance')

my_columns = [deptime, arrtime, distance, #depdelay
    year, month, dayofmonth, dayofweek, uniquecarrier, flightnum, origin, dest,
    cancelled, diverted]

# You may or may not want to include the variable 'depdelay' into
# your model, depending on whether you assume to know the delay at
# departure at the time when you are trying to estimate the delay
# at arrival.

### Instantiate linear classifier

classifier = tf.estimator.LinearClassifier(feature_columns = my_columns)

### Train linear classifier

classifier.train(train_inpf)

### Define wrapper for input function for evaluation step

eval_inpf = functools.partial(input_fn, predict_file, num_epochs=1,
    shuffle=False, batch_size=100)

### Define alternative function to calulate the area under the curve

def metric_auc(labels, predictions):
    return {
        'auc_precision_recall': tf.metrics.auc(
            labels=labels, predictions=predictions['logistic'], num_thresholds=200,
            curve='PR', summation_method='careful_interpolation')
    }

### Add new function to the classifier

classifier = tf.contrib.estimator.add_metrics(classifier, metric_auc)

### Evaluate linear classifier

result = classifier.evaluate(eval_inpf)

### Print evaluation metrics

for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))

### Define wrapper for input function for prediction step

predict_inpf = functools.partial(input_fn, predict_file, num_epochs=1,
    shuffle=False, batch_size=100)

### Print prediction results

pred_results = classifier.predict(input_fn=predict_inpf)
for i in range(10):
    print(next(pred_results))

### Print model weights

for name in model.get_variable_names():
    print(name)
    print(model.get_variable_value(name))

# Alternatively, one could also write function like the following to retrieve
# the weights...
def get_flat_weights(model):
    weight_names = [
        name for name in model.get_variable_names()
        if "linear_model" in name and "Ftrl" not in name]
    weight_values = [model.get_variable_value(name) for name in weight_names]
    weights_flat = np.concatenate([item.flatten() for item in weight_values], axis=0)
    return weight_names, weights_flat
