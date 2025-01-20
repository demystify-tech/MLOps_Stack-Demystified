# Databricks notebook source
# MAGIC %md #Installs

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md #Imports

# COMMAND ----------

import yaml
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
# Generate and Write Features Notebook
#
# This notebook can be used to generate and write features to a Databricks Feature Store table.
# It is configured and can be executed as the tasks in the write_feature_table_job workflow defined under
# ``ac_mlops_pipeline/resources/feature-engineering-workflow-resource.yml``
#
# Parameters:
#
# * feature_catalog (required)      - Source of All the Features.
# * features (required)             - A comma separated string of features which needs to be selected from feature table .
# *
# * input_country (required)       - Used to load the country specific yml config 
# * feature_schema_name(required)   - Feature Schema Name to retrieve the computed features.
# * data_transform_module (required) - Python module containing the transform logic.
##################################################################################


# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Input Catalog Name, Source of all the features .
dbutils.widgets.text(
    "feature_catalog",
    "acda-ml-staging",
    label="Input Catalog Name",
)

# Feature Schema Name to retrieve the computed features.
# The Final Train & Test data would be stored in the respective delta tables 
dbutils.widgets.text(
    "feature_schema_name",
    "acda_ml",
    label="Feature store Schema Name",
)

# Feature transform module name.
dbutils.widgets.text(
    "transform_module", "traintestsplit", label="Features transform file."
)

dbutils.widgets.text(
    "env", "dev", label="ML Environment."
)


# Primary Keys columns for the feature table;
dbutils.widgets.text(
    "preprocess_fn",
    "repurchase_preprocess_data",
    label="Function to pre-process ",
)

# Input root path of the configs.
dbutils.widgets.text("config_path", "files/common/clv/raw", label="Input config path")


# COMMAND ----------

# Retrieve the feature schema name from the widget
env = dbutils.widgets.get('env')

#Retrieve the preprocess function
preprocess_fn = dbutils.widgets.get('preprocess_fn')



# COMMAND ----------

root_path = dbutils.widgets.get("config_path") # get the use-level config root path , in this case repurchase classifier is the use-case

common_config=f"""{root_path}/common.yml"""
country_config=f"""{root_path}/{country}.yml"""

# COMMAND ----------


notebook_path =  '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../segregator

# COMMAND ----------

# MAGIC %md
# MAGIC #Functions loaded from the common utils

# COMMAND ----------

common_utils_path = notebook_path[:notebook_path.index("files")]
common_utils_path = f"{common_utils_path}files/common"
sys.path.append(common_utils_path)

# Verify the path and list files in the directory to ensure utils.py exists
import os
print(os.listdir(common_utils_path))

from utils import *
from utils import get_config_paths, yml_loader


# COMMAND ----------

# MAGIC %md
# MAGIC #Load Configs

# COMMAND ----------

yaml_root,config_path=get_config_paths('files',common_config,notebook_path)
config_values=yml_loader(config_path)

# COMMAND ----------

config_values

# COMMAND ----------

feature_schema_name = dbutils.widgets.get('feature_schema_name')
input_catalog = config["data"]["input_catalog"]
feature_catalog = config["data"]["out_catalog"]
feature_schema_name = config["data"]["out_schema"]

# COMMAND ----------

# Construct the feature table name using the catalog, schema, and country
FEATURE_TABLE_NAME = f"{feature_catalog}.{feature_schema_name}.clv_repurchase_input_dataset_{country}_train"

# Display the constructed feature table name
FEATURE_TABLE_NAME

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Initialize the Feature Engineering Client
fe = FeatureEngineeringClient()

# Read the feature table into a DataFrame using the specified feature table name
feature_df = fe.read_table(name=FEATURE_TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC #Train Test Split

# COMMAND ----------

env=feature_catalog.split("-")[2].lower() #ensure the naming standard of catalog is not changed

# COMMAND ----------

from importlib import import_module

# Get the name of the module to import from a Databricks widget
transform_module = dbutils.widgets.get("transform_module")

# Dynamically import the specified module
mod = import_module(transform_module)

# Get the 'repurchase_preprocess_data' function from the imported module
preprocess_data_fn = getattr(mod, preprocess_fn)

# Call the preprocessing function with the feature DataFrame and configuration values
# This function is expected to return the training and testing datasets
X_train, X_test, y_train, y_test = preprocess_data_fn(feature_df, config_values,env,country_config_values)

# COMMAND ----------

writeTrainTest(writer,X_train, X_test, y_train, y_test,feature_catalog,feature_schema_name,config_values['rootTableName'],country)

# COMMAND ----------

dbutils.notebook.exit(0)
