# Databricks notebook source
# MAGIC %md
# MAGIC #Installs

# COMMAND ----------

# MAGIC %pip install pyyaml
# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

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
    "metric", "recall", label="metric to evaluate"
)

# Input root path of the configs.
dbutils.widgets.text("root_path", "files/common/clv/repurchase_classifier", label="Input root path")

# COMMAND ----------

# Import the os module to interact with the operating system
import os

# Construct the notebook path by appending the directory of the current notebook to '/Workspace/'
# This uses the dbutils object to get the current notebook context and extract its path
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# Change the current working directory to the notebook path
# This is useful for relative file operations within the notebook
%cd $notebook_path

# COMMAND ----------

# MAGIC %md
# MAGIC #Functions loaded from the common utils
# MAGIC

# COMMAND ----------

# Add comments to explain the purpose of the code
# Set the common_utils_path variable to the directory path before "files" in the notebook_path
common_utils_path = notebook_path[:notebook_path.index("files")]
print(common_utils_path)

# Append "files/common" to the common_utils_path
common_utils_path = f"{common_utils_path}files/common"
sys.path.append(common_utils_path)

# Verify the path and list files in the directory to ensure utils.py exists
import os
print(os.listdir(common_utils_path))

# Import the necessary functions from the utils module
from utils import *
from utils import register_and_deploy_model

# COMMAND ----------

# MAGIC %md
# MAGIC #Load Configs

# COMMAND ----------

# Get the user-level config root path from the widget, specific to the repurchase classifier use-case
root_path = dbutils.widgets.get("root_path")

# Construct the path to the common configuration file
common_config = f"{root_path}/common.yml"

# Get the configuration paths using the get_config_paths function
yaml_root, config_path = get_config_paths('files', common_config, notebook_path)

# Load the configuration values from the YAML file
config_values = yml_loader(config_path)

# Display the loaded configuration values
config_values

# COMMAND ----------

# Get the country value from the input widget
country = dbutils.widgets.get("input_country")

# Get the training set catalog value from the input widget
training_set_catalog = dbutils.widgets.get("training_set_catalog")

# Get the training set schema name value from the input widget
training_set_schema_name = dbutils.widgets.get("training_set_schema_name")
metric = dbutils.widgets.get("metric")

# Get the rootTableName value from the config_values dictionary
rootTableName = config_values['rootTableName']

# Construct the namespace using the training set catalog, training set schema name, and rootTableName
namespace = f"`{training_set_catalog}`.{training_set_schema_name}.{rootTableName}"

# COMMAND ----------

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# Retrieve the experiment ID from the job's task values, with a default fallback if not found
experimentId=dbutils.jobs.taskValues.get('Training',"experimentId", debugValue="")

# Retrieve the current run ID from the job's task values, with a default fallback if not found
current_run_id=dbutils.jobs.taskValues.get('Training',"runId", debugValue="1fd279168efa4bcab12c7595942e9f70")

# Retrieve the run name from the job's task values, with a default fallback if not found
runName=dbutils.jobs.taskValues.get('Training',"runName", debugValue="")

# Construct the model name using the training set catalog, schema name, root table name, and country
modelName=f"{training_set_catalog}.{training_set_schema_name}.clv_{rootTableName}_{country}"

# COMMAND ----------

# Register the model with the given current_run_id and modelName
# and deploy it for serving
register_and_deploy_model(current_run_id, modelName,metric)
