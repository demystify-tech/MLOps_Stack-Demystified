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
# * input_catalog (required)      - Source of All the Features.
# * primary_keys (required)       - A comma separated string of primary key columns of the output feature table.
# *
# * root_path (required)           - Root path of the feature engineering configs.
# * input_country (required)       - Used to load the country specific yml config 
# * output_catalog_name(required)  - Feature Catalog Name to store the computed features.
# * output_schema_name(required)   - Feature Schema Name to store the computed features.
# * features_transform_module (required) - Python module containing the feature transform logic.
##################################################################################


# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Input Catalog Name, Source of all the features .
dbutils.widgets.text(
    "input_catalog",
    "overwatch_nonprod",
    label="Input Catalog Name",
)

# Input root path of the feature engineering configs.
dbutils.widgets.text("config_path", "", label="Input config path")


# Feature transform module name.
dbutils.widgets.text(
    "features_transform_module", "pickup_features", label="Features transform file."
)
# Primary Keys columns for the feature table;
dbutils.widgets.text(
    "primary_keys",
    "id",
    label="Primary keys columns for the feature table, comma separated.",
)

# COMMAND ----------

root_path = dbutils.widgets.get("config_path") # get the use-level config root path , in this case repurchase classifier is the use-case
config_path=f"""{root_path}/common.yml"""
config_path

# COMMAND ----------


notebook_path =  '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../features

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

from utils import get_config_paths, yml_loader

# COMMAND ----------

yaml_root,config_path=get_config_paths('files',config_path,notebook_path)
config=yml_loader(config_path)
feature_tbl_name =f"feature_train"

# COMMAND ----------

config

# COMMAND ----------

input_catalog = config["data"]["input_catalog"]
pk=dbutils.widgets.get('primary_keys')
features_module = dbutils.widgets.get("features_transform_module")
output_catalog_name = config["data"]["out_catalog"]

output_schema_name = config["data"]["out_schema"]
truncate_query= f"Truncate table `{output_catalog_name}`.{output_schema_name}.{feature_tbl_name}"
feature_tbl_name=f"{output_catalog_name}.{output_schema_name}.{feature_tbl_name}"

# COMMAND ----------

# DBTITLE 1,Compute features.
# Compute the features. This is done by dynamically loading the features module.
from importlib import import_module

mod = import_module(features_module)
compute_features_fn = getattr(mod, "compute_features")

features_df = compute_features_fn(config, input_catalog, pk,spark)

# features_df.display()

# COMMAND ----------

# DBTITLE 1,Write computed features.
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create the feature table if it does not exist first.
# Note that this is a no-op if a table with the same name and schema already exists.

fe.create_table(
    name=feature_tbl_name,    
    primary_keys=pk,
    df=features_df,
)

spark.sql(truncate_query)

# Write the computed features dataframe.
fe.write_table(
    name=feature_tbl_name,
    df=features_df,
)

# COMMAND ----------

dbutils.notebook.exit(0)
