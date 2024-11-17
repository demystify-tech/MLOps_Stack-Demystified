# Databricks notebook source
##################################################################################
# Model Training Notebook using RandomForest and MLflow
#
# This notebook demonstrates a training pipeline using the RandomForest algorithm.
# It is configured and can be executed as a model training task in an MLOps workflow.
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - MLflow registered model name to use for the trained model.
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# Notebook arguments (provided via widgets or notebook execution arguments)

dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-mlops-experiment",
    label="MLflow experiment name",
)

# MLflow registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", f"airbnb_pricer_training", label="Model Name"
)

dbutils.widgets.text(
    "catalog", f"overwatch_nonprod", label="Catalog Name"
)

dbutils.widgets.text(
    "schema", f"airbnb", label="Schema Name"
)


# COMMAND ----------

# DBTITLE 1, Define input variables
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
ID_COL = "ID"
LABEL_COL = "price"

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1, Sample data creation (replace with actual data)
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import sklearn
from datetime import timedelta, datetime
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql import functions as F


# Read data and add a unique id column (not mandatory but preferred)
raw_df = (spark.read.format("parquet")
  .load("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/")
  .withColumn(ID_COL, F.expr("uuid()"))
)


# Split the data into training and testing sets
features_list = ["bedrooms", "neighbourhood_cleansed", "accommodates", "cancellation_policy", "beds", "host_is_superhost", "property_type", "minimum_nights", "bathrooms", "host_total_listings_count", "number_of_reviews", "review_scores_value", "review_scores_cleanliness"]

train_df, baseline_test_df, inference_df = raw_df.select(*features_list+[ID_COL, LABEL_COL]).randomSplit(weights=[0.6, 0.2, 0.2], seed=42)

# COMMAND ----------

# DBTITLE 1, Train and log the RandomForest model using MLflow
# Define the training datasets
X_train = train_df.drop(ID_COL, LABEL_COL).toPandas()
Y_train = train_df.select(LABEL_COL).toPandas().values.ravel()

# Define categorical preprocessor
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]
one_hot_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("onehot", one_hot_pipeline, categorical_cols)], remainder="passthrough", sparse_threshold=0)

# Define the model
skrf_regressor = RandomForestRegressor(
  bootstrap=True,
  criterion="squared_error",
  max_depth=5,
  max_features=0.5,
  min_samples_leaf=0.1,
  min_samples_split=0.15,
  n_estimators=36,
  random_state=42,
)

model = Pipeline([
  ("preprocessor", preprocessor),
  ("regressor", skrf_regressor),
])


# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True, registered_model_name=model_name)

# Start MLflow run
run_name = "RandomForestRegressor_Training"
with mlflow.start_run(run_name=run_name) as run:
    
    # Train the model
    model.fit(X_train, Y_train)
    
    # Log the model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model_test"
    )
    
    # # Log the model parameters
    # mlflow.log_params(params)

    # Register the model with MLflow
    client = MlflowClient()
    # version = client.get_latest_versions(name=model_name)[0].version
    # model_uri = f"models:/{model_name}/{version}"
    model_uri = f"runs:/{run.info.run_id}/random_forest_model_test"
    model_version = mlflow.register_model(model_uri, model_name)

    print(f"Model registered as version {model_version.version} with run ID: {run.info.run_id}")

# COMMAND ----------

# DBTITLE 1, Store model URI and exit
# Store the model URI and version for later retrieval in deployment tasks
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version.version)

# Exit the notebook and pass the model URI as the output
dbutils.notebook.exit(model_uri)