# Databricks notebook source
# MAGIC %md
# MAGIC #Installs

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC %pip install databricks-feature-engineering
# MAGIC %pip install xgboost matplotlib seaborn
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
# Purpose:
# This notebook is dedicated to the training of machine learning models. It includes
# steps for loading training data, preprocessing, model training, and evaluation.
#
# Parameters:
# * training_set_catalog (required) - Identifier for the source of training and test datasets.
# * input_country (required) - Specifies the country for which the model is being trained, 
#   allowing for country-specific configurations.
# * training_set_schema_name (required) - Name of the schema where computed features are stored.
#   This schema is used to retrieve features for model training.
# * module (required) - Name of the Python module that contains the logic for model training.
#   This allows for modular training logic that can be updated independently.
# * root_path (required) - The root directory path where configuration files are stored. 
#   This is used to load necessary configurations for model training.
#
# Usage:
# This notebook is designed to be run with specific parameters that guide the training process.
# These parameters can be provided through Databricks widgets or notebook arguments, enabling
# dynamic execution based on the input parameters.
##################################################################################

# Define Databricks widgets for input parameters. These widgets allow users to input or select
# the necessary parameters for model training directly within the Databricks UI.

# Input Catalog Name: Source of all the features for training.
dbutils.widgets.text(
    "training_set_catalog",
    "acda-ml-staging",
    label="Input Catalog Name",
)

# Input Country: Specifies the country for which the model is being trained.
dbutils.widgets.text("input_country", "DE", label="Input Country")

# Training Set Schema Name: Name of the schema to retrieve computed features for training.
dbutils.widgets.text(
    "training_set_schema_name",
    "acda_ml",
    label="Training set store Schema Name",
)

# Module: Name of the Python module containing the model training logic.
dbutils.widgets.text(
    "model", "log_reg", label="model function to train"
)

# Root Path: The root directory path for configuration files related to model training.
dbutils.widgets.text("root_path", "files/common/clv/repurchase_classifier", label="Input root path")

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %md
# MAGIC #Functions loaded from the common utils
# MAGIC

# COMMAND ----------

common_utils_path = notebook_path[:notebook_path.index("files")]
print(common_utils_path)
common_utils_path = f"{common_utils_path}files/common"
sys.path.append(common_utils_path)

# Verify the path and list files in the directory to ensure utils.py exists
import os
print(os.listdir(common_utils_path))

from utils import *

# COMMAND ----------

# MAGIC %md
# MAGIC #Load Configs
# MAGIC

# COMMAND ----------

root_path = dbutils.widgets.get("root_path") # get the use-level config root path , in this case repurchase classifier is the use-case

common_config=f"""{root_path}/common.yml"""
yaml_root,config_path=get_config_paths('files',common_config,notebook_path)
config_values=yml_loader(config_path)
config_values

# COMMAND ----------

# Retrieve the input country from the widget
country = dbutils.widgets.get("input_country")

#Retrieve the model name 
model = dbutils.widgets.get("model")

# Retrieve the training set catalog from the widget
training_set_catalog = dbutils.widgets.get("training_set_catalog")

# Retrieve the training set schema name from the widget
training_set_schema_name = dbutils.widgets.get("training_set_schema_name")

# Construct the experiment path from the notebook path
experimentPath = notebook_path[:notebook_path.index(".bundle")]

# Construct the experiment name using the experiment path and configuration values
experimentName = f"{experimentPath}{config_values['experimentName']}"

# Replace "/Workspace/" in the experiment name
experimentName = experimentName.replace("/Workspace/", "")

# Append the country to the experiment name
experimentName = f"{experimentName}_{country}"

# Retrieve the root table name from the configuration values
rootTableName = config_values['rootTableName']

# Construct the namespace using the training set catalog, schema name, and root table name
namespace = f"`{training_set_catalog}`.{training_set_schema_name}.{rootTableName}"

# Construct the training and testing dataset table names for the input country
X_Train = f"{namespace}_x_train_{country.lower()}"
y_Train = f"{namespace}_y_train_{country.lower()}"
X_Test  = f"{namespace}_x_test_{country.lower()}"
y_test = f"{namespace}_y_test_{country.lower()}"

# COMMAND ----------

# Load the training and testing datasets into Spark DataFrames and Pandas Series
X_Train_df, y_Train_series, X_Test_df, y_Test_series = load_train_test_data(X_Train, y_Train, X_Test, y_test, spark)

# Set the MLflow experiment using the constructed experiment name
exp = mlflow.set_experiment(experimentName)

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# Import the import_module function from the importlib module
from importlib import import_module

# Dynamically import the specified module
mod = import_module("utils")

# Get the preprocess_inference_data function from the imported module
model_fn = getattr(mod, model)

# COMMAND ----------

# Train the model and log the run details using the logistic regression function
model, run_name, run_id = model_fn(X_Train_df, y_Train_series, X_Test_df, y_Test_series, exp.experiment_id, country, config_values)

# COMMAND ----------

# Extract the run ID, experiment ID, and run name from the run information
(run_id.info.run_id, run_id.info.experiment_id, run_id.info.run_name)

# COMMAND ----------

# Set the experiment ID, run ID, and run name as task values for downstream tasks
dbutils.jobs.taskValues.set("experimentId", run_id.info.experiment_id)
dbutils.jobs.taskValues.set("runId", run_id.info.run_id)
dbutils.jobs.taskValues.set("runName", run_id.info.run_name)
