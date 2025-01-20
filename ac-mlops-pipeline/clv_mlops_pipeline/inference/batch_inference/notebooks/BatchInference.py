# Databricks notebook source
# MAGIC %md
# MAGIC #Installs

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC %pip install databricks-feature-engineering
# MAGIC %pip install xgboost matplotlib seaborn
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #Imports

# COMMAND ----------

import yaml
import mlflow
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pyspark.pandas as ps
from pyspark.sql.functions import monotonically_increasing_id
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

# COMMAND ----------

##################################################################################
# Model Training Notebook
#
# This notebook is used for Model Training
#
# Parameters:
#
# * training_set_catalog (required)      - Source of All the Train Test Dataset.
# * input_country (required)             - Used to load the country specific yml config 
# * training_set_schema_name (required)  - Feature Schema Name to retrieve the computed features.
# * module (required)                    - Python module containing the model logic.
##################################################################################

# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Input Catalog Name, Source of all the features.
dbutils.widgets.text(
    "training_set_catalog",
    "acda-ml-staging",
    label="Input Catalog Name",
)

# Input Country
dbutils.widgets.text("input_country", "DE", label="Input Country")

# Feature Schema Name to retrieve the computed features.
# The Final Train & Test data would be stored in the respective delta tables.
dbutils.widgets.text(
    "training_set_schema_name",
    "acda_ml",
    label="Training set store Schema Name",
)

# Feature transform module name.
dbutils.widgets.text(
    "module", "inferenceutils", label="module to load inference pre-process module"
)

# Feature transform module name.
dbutils.widgets.text(
    "preprocess_fn", "preprocess_inference_data", label="function to pre-process"
)

# Input root path of the configs.
dbutils.widgets.text("root_path", "files/common/clv/repurchase_classifier", label="Input root path")

# COMMAND ----------

import os

# Get the current notebook path
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# Change directory to the notebook path
os.chdir(notebook_path)

# Change directory to the batchutils directory
os.chdir("../batchutils")

# COMMAND ----------

# MAGIC %md
# MAGIC #Functions loaded from the common utils
# MAGIC

# COMMAND ----------

# Extract the common utilities path from the notebook path
common_utils_path = notebook_path[:notebook_path.index("files")]

# Print the common utilities path for verification
print(common_utils_path)

# Append the common utilities path to the system path
common_utils_path = f"{common_utils_path}files/common"
sys.path.append(common_utils_path)

# Verify the path and list files in the directory to ensure utils.py exists
import os
print(os.listdir(common_utils_path))

# Import all functions and classes from utils.py
from utils import *

# COMMAND ----------

# Import the import_module function from the importlib module
from importlib import import_module

# Get the module name from the widget
module = dbutils.widgets.get("module")

# Dynamically import the specified module
mod = import_module(module)

preprocess_fn = dbutils.widgets.get("preprocess_fn")


# Get the preprocess_inference_data function from the imported module
preprocess_inference_data_fn = getattr(mod, preprocess_fn)

# Display the features DataFrame
# features_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #Load Configs
# MAGIC

# COMMAND ----------

# Retrieve the root path configuration from the widget, specific to the use-case (e.g., repurchase classifier)
root_path = dbutils.widgets.get("root_path")

# Construct the path to the common configuration file using the root path
common_config = f"{root_path}/common.yml"

# Utilize the get_config_paths function to obtain the root of the YAML configuration and the specific configuration path
yaml_root, config_path = get_config_paths('files', common_config, notebook_path)

# Load the configuration values from the specified YAML configuration file
config_values = yml_loader(config_path)

# Display the loaded configuration values
config_values

# COMMAND ----------

# Retrieve the input country from the widget
country = dbutils.widgets.get("input_country")

# Retrieve the training set catalog from the widget
training_set_catalog = dbutils.widgets.get("training_set_catalog")

# Retrieve the training set schema name from the widget
training_set_schema_name = dbutils.widgets.get("training_set_schema_name")

# Retrieve the root table name from the configuration values
rootTableName = config_values['rootTableName']

# Construct the namespace using the training set catalog, schema name, and root table name
namespace = f"`{training_set_catalog}`.{training_set_schema_name}.{rootTableName}"

# Define the primary key columns to retain
primary_key_columns = config_values['keycolumns']

# COMMAND ----------

# Set the registry URI for MLflow to use Databricks Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# Construct the model name using the training set catalog, schema name, root table name, and country
modelName = f"{training_set_catalog}.{training_set_schema_name}.clv_{rootTableName}_{country}"

# Define the features to be dropped from the model
dropFeatures = config_values['dropFeatures']

# Retrieve the input country from the widget and construct the feature table name
FEATURE_TABLE_NAME = f"{training_set_catalog}.{training_set_schema_name}.clv_repurchase_input_dataset_{country}_train"

# Construct the training dataset name using the namespace and country, then retrieve its columns as features
X_Train = f"{namespace}_x_train_{country.lower()}"
features = spark.table(X_Train).columns

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Initialize the Feature Engineering Client
fe = FeatureEngineeringClient()

# Read the feature table into a DataFrame using the specified feature table name
inference_df = fe.read_table(name=FEATURE_TABLE_NAME)

# Filter the DataFrame to include only rows where first_purchase_made is 1, repurchase_window_passed is 0, and repurchase_flag is 0
inference_df = inference_df.filter(
    (inference_df.first_purchase_made == 1) & 
    (inference_df.repurchase_window_passed == 0) & 
    (inference_df.repurchase_flag == 0)
)

# COMMAND ----------

# Preprocess the inference data using the provided function, configuration values, primary key columns, and Spark session
pre_processed_inference_df = preprocess_inference_data_fn(inference_df, config_values,primary_key_columns,spark)

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from pyspark.sql.types import *

# Load the model using the updated registry URI
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f'models:/{modelName}@champion', result_type=ArrayType(StringType()))

from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Get the model version tagged as "Champion"
version = client.get_model_version_by_alias(modelName, "Champion")

# COMMAND ----------

if 'index' in features:
    features.remove('index')

# COMMAND ----------

# Import necessary functions from pyspark.sql.functions
from pyspark.sql.functions import struct, lit

# Construct a struct of features to be used as input for the UDF
udf_inputs = struct(*features)

# Apply the UDF to the pre_processed_inference_df DataFrame to generate predictions
# The predictions are added as a new column named 'prediction'
batch_inference_data = pre_processed_inference_df.withColumn(
    'prediction',
    apply_model_udf(udf_inputs)
)

# COMMAND ----------


# COMMAND ----------

# usage of the overwrite_with_cdf function
# This function overwrites the existing data in the specified table with the new batch inference data and stores in delta table
overwrite_with_cdf(batch_inference_data, f"{namespace}_{country}_inference")
