# Databricks notebook source
# MAGIC %md
# MAGIC ## Extract Model Metrics
# MAGIC
# MAGIC ### Job Logic:
# MAGIC 1. Use MLFLOW APIs to extract the model details.
# MAGIC 2. Extract the relevant metrics and persist in a Delta table.
# MAGIC 3. Perform SCD Type 2 for incremental records to capture Challenger/Champion metrics over time.
# MAGIC
# MAGIC **Note:** If a model metric is not available through `mlflow.autolog()`, the user needs to retrospect and log the metric using MLFLOW APIs.
# MAGIC
# MAGIC ### Available Columns:
# MAGIC - **Run ID**: String - Primary Key
# MAGIC - **Experiment ID**: String
# MAGIC - **Run Name**: String
# MAGIC - **Alias Type**: String - Challenger/Champion
# MAGIC - **Model Name**: String
# MAGIC - **Ingestion Time**: Timestamp
# MAGIC - **Metrics**: JSON  - Can contain multiple metrics
# MAGIC - **Params**: JSON  - Model parameters
# MAGIC - **Notebook Details**: JSON 
# MAGIC - **Profile Metrics**: JSON  - Feature shape, size, etc.
# MAGIC - **Start Time**: Timestamp - SCD Column
# MAGIC - **End Time**: Timestamp - SCD Column
# MAGIC - **Active Flag**: String - SCD Column
# MAGIC - **Version**: String

# COMMAND ----------

import mlflow
from datetime import datetime
from delta.tables import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Create widgets for user input to capture catalog names, catalog name, and schema name
dbutils.widgets.text("model_catalogs", "Provide catalog names where model is registered like 'acda-ml-staging', acda-ml-prod")
dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")

# Retrieve widget values entered by the user
model_catalogs = dbutils.widgets.get("model_catalogs")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

# Split the model_catalogs values by comma and convert to a list
catalog_names = model_catalogs.split(',')

# Construct the table name using catalog and schema names
table_name = f"`{catalog_name}`.{schema_name}.model_metrics"

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Initialize the MLflow client to interact with the MLflow tracking server
client = mlflow.MlflowClient()

# COMMAND ----------

# Extract all registered models from Unity Catalog

def get_all_registered_models(catalog_names):
    """
    Retrieve all registered models from MLflow Unity Catalog and return as a list of model names.

    Parameters:
    catalog_names (list): A list of catalog names to filter the registered models.

    Returns:
    list: A list containing names of registered models.
    """
    import mlflow
    import re
    from pyspark.sql.functions import col

    # Initialize MLflow client
    client = mlflow.MlflowClient()

    # Get all registered models
    registered_models = client.search_registered_models()

    # Filter models based on catalog names using regular expressions
    filtered_models = [model.name for model in registered_models if any(re.search(catalog.strip(), model.name) for catalog in catalog_names)]

    return filtered_models

# Display the DataFrame with all registered models
models = get_all_registered_models(catalog_names)
models

# COMMAND ----------

def get_model_details(model_name, alias):
    """
    Retrieve model details from MLflow and return as a Spark DataFrame.

    Parameters:
    model_name (str): The name of the model.
    alias (str): The alias of the model version (e.g., "Champion" or "Challenger").

    Returns:
    DataFrame: A Spark DataFrame containing model details.
    """
    from datetime import datetime
    import mlflow
    from pyspark.sql.functions import col, lit, from_json, to_date
    from pyspark.sql.types import StringType, MapType

    # Initialize MLflow client
    client = mlflow.MlflowClient()
    
    # Get model version details using alias
    mod = client.get_model_version_by_alias(model_name, alias)
    
    # Convert timestamps to human-readable format
    created_timestamp = str(datetime.fromtimestamp(mod.creation_timestamp / 1000.0))
    last_updated_timestamp = str(datetime.fromtimestamp(mod.last_updated_timestamp / 1000.0))
    
    # Set active flag
    active_flag = "Y"
    
    # Get run details
    run_id = mod.run_id
    run = client.get_run(run_id)
    version = mod.version

    # Define columns for the DataFrame
    columns = ["run_id", "experiment_id", "run_name", "alias", "model_name", "version", 
               "created_timestamp", "last_updated_timestamp", "ingestion_time", 
               "metrics", "params", "dataset_profile", "dataset_source"]

    # Prepare data for the DataFrame
    data = [(run.info.run_id, run.info.experiment_id, run.info.run_name, alias, model_name, version, 
             created_timestamp, last_updated_timestamp, datetime.now(), run.data.metrics, 
             run.data.params, run.inputs.dataset_inputs[0].dataset.profile, 
             run.inputs.dataset_inputs[0].dataset.source)]

    # Create Spark DataFrame
    df = spark.createDataFrame(data, columns)
    
    # Parse JSON string in 'dataset_profile' column to a MapType
    df = df.withColumn("dataset_profile", from_json(col("dataset_profile"), MapType(StringType(), StringType())))
    
    # Add 'active_flag' column
    df = df.withColumn("active_flag", lit(active_flag))
    
    # Add 'start_date' column
    df = df.withColumn("start_date", to_date(col("created_timestamp")))
    
    # Add 'end_date' column with None value
    df = df.withColumn("end_date", lit(None).cast(StringType()))
    
    return df

# COMMAND ----------

from mlflow.exceptions import RestException

masterDF = None
for model in models:
    try:
        unionDF = get_model_details(model, "Champion").unionAll(
            get_model_details(model, "Challenger")
        )
        if masterDF is None:
            masterDF = unionDF
        else:
            masterDF = masterDF.unionAll(unionDF)
    except RestException as e:
        print(f"Error retrieving run details for model {model}: {e}")

display(masterDF)

# COMMAND ----------

# Check if the table exists in the Spark catalog
if spark.catalog.tableExists(table_name) == False:
    # Create the table with the schema derived from masterDF
    create_table_sql = f"CREATE TABLE {table_name} (id BIGINT GENERATED BY DEFAULT AS IDENTITY, {', '.join([f'{field.name} {field.dataType.simpleString()}' for field in masterDF.schema])})"
    spark.sql(create_table_sql)
    
    # Write the masterDF DataFrame to the table with schema merging enabled
    masterDF.write.option("mergeSchema", "true").mode("overwrite").saveAsTable(table_name)
    
    # Exit the notebook as only this transaction is applicable for the first run
    dbutils.notebook.exit("For the first run, only this transaction is applicable, rest of the notebook is not applicable")

# COMMAND ----------

spark.table(table_name).display()

# COMMAND ----------

masterDF = masterDF.withColumn("id", lit(None).cast("bigint"))
masterDF = masterDF.withColumn("run_name", lit("XGB_Classification_DE")) #need to be removed
masterDF = masterDF.select(['id'] + [col for col in masterDF.columns if col not in {'start_date', 'end_date', 'id'}])
display(masterDF)

# COMMAND ----------

# MAGIC %md ### Merge tables
# MAGIC <p>Insert if new, Update if already exists</p>

# COMMAND ----------

scdType2DF=spark.table(table_name)

# COMMAND ----------

# Create list of selected employee_id's
runList = masterDF.select(collect_list(masterDF['run_id'])).collect()[0][0]

# Select columns in new dataframe to merge
scdChangeRows = masterDF.selectExpr(
  "null AS id", "run_id", "experiment_id", "run_name", "alias", "model_name","version", "created_timestamp", "last_updated_timestamp", "ingestion_time", "metrics","params","dataset_profile","dataset_source","active_flag", "current_date AS start_date", "null AS end_date"
)

# Union join queries to match incoming rows with existing
scdChangeRows = scdChangeRows.unionByName(
  scdType2DF
  .where(col("run_id").isin(runList)), allowMissingColumns=True
)
# Preview results
display(scdChangeRows)

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql.functions import current_date, lit

# Convert table to Delta
deltaTable = DeltaTable.forName(spark, table_name)

# Merge Delta table with new dataset
(
  deltaTable
    .alias("original2")
    # Merge using the following conditions
    .merge(
      scdChangeRows.alias("updates2"),
      "original2.id = updates2.id"
    )
    # When matched UPDATE ALL values
    .whenMatchedUpdate(
      set={
        "end_date": current_date(),
        "active_flag": lit("N"),
      }
    )
    # When not matched INSERT ALL rows
    .whenNotMatchedInsertAll()
    # Execute
    .execute()
)

# COMMAND ----------

spark.table(table_name).display()

# COMMAND ----------

spark.table(table_name).selectExpr("ROW_NUMBER() OVER (ORDER BY id NULLS LAST) - 1 AS id",
  "run_id", "experiment_id", "run_name", "alias", "model_name","version", "created_timestamp", "last_updated_timestamp", "ingestion_time", "metrics", "params", "dataset_profile", "dataset_source", "active_flag", "start_date", "end_date"
).write.insertInto(tableName=table_name, overwrite=True)

# COMMAND ----------

spark.table(table_name).display()