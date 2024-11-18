# Databricks notebook source
# MAGIC %pip install "databricks-sdk>=0.28.0"

# COMMAND ----------

# This step is necessary to reset the environment with our newly installed wheel.
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("baseline_table", "", label="Baseline Table Name")

dbutils.widgets.text("inference_table", "", label="Inference Table Name")
dbutils.widgets.text(
    "model_name", "dev-my_mlops_stack-model", label="Model Name"
)
dbutils.widgets.text(
    "prediction_col", "prediction", label="Prediction Column"
)
dbutils.widgets.text(
    "label", "price", label="Label Column"
)
dbutils.widgets.text(
    "id_col", "ID", label="Label Column"
)
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")

dbutils.widgets.text(
    "new_observation", "overwatch_nonprod.airbnb.airbnb_simulated", label="Simulated Data"
)

# COMMAND ----------

BASELINE_TABLE = dbutils.widgets.get("baseline_table")
INFERENCE_TABLE = dbutils.widgets.get("inference_table")
MODEL_NAME = dbutils.widgets.get("model_name")
PREDICTION_COL = dbutils.widgets.get("prediction_col")
LABEL_COL = dbutils.widgets.get("label")
ID_COL = dbutils.widgets.get("id_col")
TIMESTAMP_COL = "timestamp"
MODEL_ID_COL = "model_id"
env = dbutils.widgets.get("env")
OBSERVED_DATA = dbutils.widgets.get("new_observation")

# COMMAND ----------

import mlflow
import sklearn

from datetime import timedelta, datetime
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql import functions as F
from pyspark.sql.functions import struct, lit, to_timestamp

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create Baseline Table

# COMMAND ----------
def get_deployed_model_stage_for_env(env):
    """
    Get the model version stage under which the latest deployed model version can be found
    for the current environment
    :param env: Current environment
    :return: Model version stage
    """
    # For a registered model version to be served, it needs to be in either the Staging or Production
    # model registry stage
    # (https://learn.microsoft.com/azure/databricks/machine-learning/manage-model-lifecycle/workspace-model-registry).
    # For models in dev and staging environments, we deploy the model to the "Staging" stage, and in prod we deploy to the
    # "Production" stage
    _MODEL_STAGE_FOR_ENV = {
        "dev": "Staging",
        "staging": "Staging",
        "prod": "Production",
        "test": "Production",
    }
    return _MODEL_STAGE_FOR_ENV[env]

stage = get_deployed_model_stage_for_env(env)

model_uri = f"models:/{MODEL_NAME}/{stage}"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
features = ["bedrooms", "neighbourhood_cleansed", "accommodates", "cancellation_policy", "beds", "host_is_superhost", "property_type", "minimum_nights", "bathrooms", "host_total_listings_count", "number_of_reviews", "review_scores_value", "review_scores_cleanliness"]

# COMMAND ----------

model_version_infos = MlflowClient().search_model_versions("name = '%s'" % MODEL_NAME)
model_version = max(
    int(version.version)
    for version in model_version_infos
    if version.current_stage == stage
)

# COMMAND ----------

from pyspark.sql.functions import struct, lit, to_timestamp


baseline_test_df = spark.read.table("overwatch_nonprod.airbnb.baseline_test")
prediction_df = baseline_test_df.withColumn("prediction", loaded_model(struct([baseline_test_df[col] for col in features])))

baseline_test_df_with_pred = prediction_df.withColumn(MODEL_ID_COL, F.lit(model_version))

display(baseline_test_df_with_pred)

# COMMAND ----------

# BASELINE_TABLE = f"overwatch_nonprod.airbnb.{BASELINE_TABLE_NAME}"

# COMMAND ----------

if spark.catalog.tableExists(BASELINE_TABLE):
  pass
else:
  (baseline_test_df_with_pred
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable(BASELINE_TABLE)
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create Inference Table

# COMMAND ----------

inference_df = spark.read.table(INFERENCE_TABLE).select(*features+[ID_COL, LABEL_COL])

# COMMAND ----------

test_labels_df = inference_df.select(ID_COL, LABEL_COL)


# COMMAND ----------

# Simulate timestamp(s) if they don't exist
timestamp1 = (datetime.now() + timedelta(1)).timestamp()

pred_df1 = (inference_df
  .withColumn(TIMESTAMP_COL, F.lit(timestamp1).cast("timestamp")) 
  .withColumn(PREDICTION_COL, loaded_model(struct([inference_df[col] for col in features])))
)

# COMMAND ----------

TABLE_NAME = INFERENCE_TABLE + "__inference"

if spark.catalog.tableExists(TABLE_NAME):
  pass
else:
  (pred_df1
    .withColumn(MODEL_ID_COL, F.lit(model_version))
    .withColumn(LABEL_COL, F.lit(None).cast("double"))
    .write.format("delta").mode("overwrite") 
    .option("mergeSchema",True) 
    .option("delta.enableChangeDataFeed", "true") 
    .saveAsTable(TABLE_NAME)
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM overwatch_nonprod.airbnb.dev_airbnb_predictions_inference;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5. Create the monitor

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorInfoStatus, MonitorRefreshInfoState, MonitorMetric

w = WorkspaceClient()

# COMMAND ----------

# ML problem type, one of "classification"/"regression"
PROBLEM_TYPE = MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION

# Window sizes to analyze data over
GRANULARITIES = ["1 day"]                       

# Optional parameters to control monitoring analysis. 
SLICING_EXPRS = ["cancellation_policy", "accommodates > 2"]  # Expressions to slice data with

# Directory to store generated dashboard
ASSETS_DIR = f"/Workspace/Users/sourav.banerjee@databricks.com//databricks_lakehouse_monitoring/{TABLE_NAME}"

# COMMAND ----------

CATALOG = "overwatch_nonprod"
SCHEMA = "airbnb"

# COMMAND ----------

help(w.quality_monitors)

# COMMAND ----------

try:
  w.quality_monitors.get(table_name=TABLE_NAME)
except:
  print("Monitor not found")
  print(f"Creating monitor for {TABLE_NAME}")

  info = w.quality_monitors.create(
    table_name=TABLE_NAME,
    inference_log=MonitorInferenceLog(
      granularities=GRANULARITIES,
      timestamp_col=TIMESTAMP_COL,
      model_id_col=MODEL_ID_COL, # Model version number 
      prediction_col=PREDICTION_COL,
      problem_type=PROBLEM_TYPE,
      label_col=LABEL_COL # Optional
    ),
    baseline_table_name=BASELINE_TABLE,
    slicing_exprs=SLICING_EXPRS,
    output_schema_name=f"{CATALOG}.{SCHEMA}",
    assets_dir=ASSETS_DIR
  )

# COMMAND ----------

import time

while info.status ==  MonitorInfoStatus.MONITOR_STATUS_PENDING:
  info = w.quality_monitors.get(table_name=TABLE_NAME)
  time.sleep(10)

assert info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"

# COMMAND ----------

# A metric refresh will automatically be triggered on creation
refreshes = w.quality_monitors.list_refreshes(table_name=TABLE_NAME).refreshes
assert(len(refreshes) > 0)

run_info = refreshes[0]
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.get_refresh(table_name=TABLE_NAME, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert(run_info.state == MonitorRefreshInfoState.SUCCESS)

# COMMAND ----------

# Display profile metrics table
profile_table = f"{TABLE_NAME}_profile_metrics"
display(spark.sql(f"SELECT * FROM {profile_table}"))

# COMMAND ----------

# Display the drift metrics table
drift_table = f"{TABLE_NAME}_drift_metrics"
display(spark.sql(f"SELECT * FROM {drift_table}"))

# COMMAND ----------

run_info = w.quality_monitors.run_refresh(table_name=TABLE_NAME)
while run_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
  run_info = w.quality_monitors.get_refresh(table_name=TABLE_NAME, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert(run_info.state == MonitorRefreshInfoState.SUCCESS)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Drift
# MAGIC

# COMMAND ----------

scoring_df2_simulated = spark.read.table(OBSERVED_DATA)

timestamp2 = (datetime.now() + timedelta(2)).timestamp()
pred_df2 = (scoring_df2_simulated
  .withColumn(TIMESTAMP_COL, F.lit(timestamp2).cast("timestamp")) 
  .withColumn(PREDICTION_COL, loaded_model(struct([scoring_df2_simulated[col] for col in features])))
  .withColumn(MODEL_ID_COL, F.lit(model_version))
  .write.format("delta").mode("append")
  .saveAsTable(TABLE_NAME)
)
