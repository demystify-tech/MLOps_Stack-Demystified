

def overwrite_with_cdf(df, tablename):
    """
    Write the DataFrame in overwrite mode with change data feed enabled.

    Parameters:
    df (DataFrame): The Spark DataFrame to be written.
    tablename (str): The name of the table where the DataFrame will be saved.

    Returns:
    None
    """
    # Write the DataFrame in overwrite mode with change data feed enabled
    df.write.format("delta") \
        .option("mergeSchema", "true") \
        .option("overwriteSchema", "true") \
        .mode("overwrite") \
        .saveAsTable(tablename)

def register_and_deploy_model(current_run_id, modelName, metric):
    """
    Registers and deploys a model in MLflow. If no model exists, it registers the first model as both Champion and Challenger.
    If a model exists, it compares the recall metric of the current run with the Champion model and updates the aliases accordingly.

    Parameters:
    current_run_id (str): The run ID of the current model training run.
    modelName (str): The name of the model to be registered and deployed.
    metric (str): The name of the metric to compare for model deployment.

    Returns:
    None
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    import time

    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % modelName)

    if len(model_version_infos) == 0:  # Registering the first model as Champion Model
        model_version = mlflow.register_model(f"runs:/{current_run_id}/model", modelName)
        client.set_registered_model_alias(modelName, "Champion", model_version.version)
        client.set_registered_model_alias(modelName, "Challenger", model_version.version)
        # Registering the model takes a few seconds, so adding a small delay
        time.sleep(15)
    elif getMetric_currenrun(current_run_id, metric) > getMetric(modelName, metric, "Champion"):
        version = client.get_model_version_by_alias(modelName, "Champion")
        model_version = mlflow.register_model(f"runs:/{current_run_id}/model", modelName)
        client.delete_registered_model_alias(modelName, "Champion")
        client.delete_registered_model_alias(modelName, "Challenger")
        client.set_registered_model_alias(modelName, "Champion", model_version.version)
        client.set_registered_model_alias(modelName, "Challenger", version.version)
    else:
        print("no eligible model to deploy")

def get_latest_model_version(model_name):
    """
    Get the latest version number of a registered model.

    Parameters:
    model_name (str): The name of the registered model.

    Returns:
    int: The latest version number of the model.
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([int(model_version_info.version) for model_version_info in model_version_infos])

def getMetric(modelname, metric, aliasName):
    """
    Retrieve a specific metric value for a given model version alias.

    Parameters:
    modelname (str): The name of the registered model.
    metric (str): The name of the metric to retrieve.
    aliasName (str): The alias of the model version (e.g., 'Champion').

    Returns:
    float: The value of the specified metric.
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    version = client.get_model_version_by_alias(modelname, aliasName)
    run_id = version.run_id
    return client.get_run(run_id).data.metrics[metric]

def getMetric_currenrun(run_id, metric):
    """
    Retrieve a specific metric value for the current run.

    Parameters:
    run_id (str): The run ID of the current run.
    metric (str): The name of the metric to retrieve.

    Returns:
    float: The value of the specified metric.
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    return client.get_run(run_id).data.metrics[metric]

def train_and_log_xgboost_model(X_Train_df, y_Train_series, X_Test_df, y_Test_series, exp, country,config_values):
    """
    Train an XGBoost model and log the run details using MLflow.

    Parameters:
    X_Train_df (DataFrame): Training features DataFrame.
    y_Train_series (Series): Training labels Series.
    X_Test_df (DataFrame): Testing features DataFrame.
    y_Test_series (Series): Testing labels Series.
    exp (mlflow.entities.Experiment): MLflow experiment object.
    config_values (dict): Configuration values for the model parameters.
    country (str): Country name for run naming and plot titles.

    Returns:
    tuple: Trained model, run name, and run ID.
    """
    import xgboost as xgb
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
    import mlflow

    # Start an MLflow experiment
    mlflow.autolog()

    run_name = f"XGB_Classification_{country}"
    with mlflow.start_run(experiment_id=exp, run_name=run_name):
        run = mlflow.active_run()
        # Use the DMatrix data structure for optimized computation
        dtrain = xgb.DMatrix(X_Train_df, label=y_Train_series)
        dtest = xgb.DMatrix(X_Test_df, label=y_Test_series)

        # Set up the parameters for XGBoost
        params = {
            'max_depth': config_values['model_params']['max_depth'],
            'eta': config_values['model_params']['eta'],
            'objective': config_values['model_params']['objective'],
            'num_class': config_values['model_params']['num_class']
        }

        # Train the model
        num_rounds = config_values['model_params']['num_rounds']
        clf = xgb.train(params, dtrain, num_rounds)

        # Predict using the trained model
        y_pred = clf.predict(dtest).argmax(axis=1)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_Test_series, y_pred)
        precision = precision_score(y_Test_series, y_pred, average='weighted')
        recall = recall_score(y_Test_series, y_pred, average='weighted')
        f1 = f1_score(y_Test_series, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_Test_series, clf.predict(dtest), multi_class='ovr')

        # Log evaluation metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)

        # Plot the prediction and actual distribution
        plt.figure()
        pd.Series(y_pred).hist(alpha=0.6, label='Prediction')
        pd.Series(y_Test_series).hist(alpha=0.6, label='Actual')
        plt.legend()
        plt.savefig("/tmp/prediction_vs_actual.png")
        mlflow.log_artifact("/tmp/prediction_vs_actual.png")

        # Compute the confusion matrix
        cm = confusion_matrix(y_Test_series, y_pred)

        # Calculate accuracy for each class
        class_recalls = cm.diagonal() / cm.sum(axis=1)
        class_precisions = cm.diagonal() / cm.sum(axis=0)

        # Log class-wise metrics to MLflow
        for i, recall in enumerate(class_recalls):
            mlflow.log_metric(f"class_{i}_recall", recall)
        for i, precision in enumerate(class_precisions):
            mlflow.log_metric(f"class_{i}_precision", precision)

        # Plot Confusion Matrix
        class_names = y_Train_series.unique().tolist()  # name of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title(f'Confusion matrix {country}', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig("/tmp/confusion_matrix.png")
        mlflow.log_artifact("/tmp/confusion_matrix.png")
        
        # Display the plot
        plt.show()
        run_id = mlflow.get_run(run.info.run_id)

        print("Model trained and logged with recall:", recall)
        return (clf, run_name, run_id)
    

def log_reg(X_Train_df, y_Train_series, X_Test_df, y_Test_series, exp, country,config_values=None):
    """
    Train and log a logistic regression model using MLflow.

    Parameters:
    X_Train_df (pd.DataFrame): Training feature set.
    y_Train_series (pd.Series): Training labels.
    X_Test_df (pd.DataFrame): Test feature set.
    y_Test_series (pd.Series): Test labels.
    exp (str): MLflow experiment ID.
    country (str): Country identifier for the run name and model logging.

    Returns:
    tuple: Trained logistic regression model, run name, and run ID.
    """
    import mlflow
    # Start an MLflow experiment
    mlflow.autolog()
    run_name = f"logistic_regression_{country}"
    with mlflow.start_run(experiment_id=exp, run_name=run_name):
        run = mlflow.active_run()
        # Train a logistic regression model
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, recall_score
        model = LogisticRegression(max_iter=10000, class_weight='balanced')
        model.fit(X_Train_df, y_Train_series)

        # Predict on test data
        predictions = model.predict(X_Test_df)
        acc = accuracy_score(y_Test_series, predictions)
        recall = recall_score(y_Test_series, predictions)

        # Log parameters and metrics
        mlflow.log_param("max_iter", 10000)
        mlflow.log_param("class_weight", 'balanced')
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)

        # Log the model
        mlflow.sklearn.log_model(model, f"logistic_regression_model_{country}")
        run_id = mlflow.get_run(run.info.run_id)

        print("Model trained and logged with recall:", recall)
        return (model, run_name, run_id)

def load_train_test_data(X_Train, y_Train, X_Test, y_test, spark):
    """
    Load training and testing data from specified table names and convert them to Pandas DataFrames and Series.

    Parameters:
    X_Train (str): The table name for the training features.
    y_Train (str): The table name for the training labels.
    X_Test (str): The table name for the testing features.
    y_test (str): The table name for the testing labels.
    spark (SparkSession): The Spark session object.

    Returns:
    tuple: A tuple containing:
        - X_Train_df (pd.DataFrame): The training features as a Pandas DataFrame.
        - y_Train_series (pd.Series): The training labels as a Pandas Series.
        - X_Test_df (pd.DataFrame): The testing features as a Pandas DataFrame.
        - y_Test_series (pd.Series): The testing labels as a Pandas Series.
    """
    import pandas as pd
    X_Train_df = spark.read.table(X_Train).toPandas().set_index("index")
    y_Train_df = spark.read.table(y_Train).toPandas().set_index("index")
    y_Train_series = pd.Series(y_Train_df.iloc[:, 0])
    X_Test_df = spark.read.table(X_Test).toPandas().set_index("index")
    y_Test_df = spark.read.table(y_test).toPandas().set_index("index")
    y_Test_series = pd.Series(y_Test_df.iloc[:, 0])
    return X_Train_df, y_Train_series, X_Test_df, y_Test_series

def replace_spaces_in_columns(df):
    """
    Replace spaces in column names with underscores.

    Parameters:
    df (pd.DataFrame): The DataFrame with columns to be renamed.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    df.columns = [col.replace(' ', '_') for col in df.columns]
    return df

def spark_replace_spaces_in_columns(df):
    """
    Replace spaces in column names with underscores for a Spark DataFrame.

    Parameters:
    df (DataFrame): The Spark DataFrame with columns to be renamed.

    Returns:
    DataFrame: The Spark DataFrame with renamed columns.
    """
    new_column_names = [col.replace(' ', '_') for col in df.columns]
    return df.toDF(*new_column_names)

def writer(df, tblname):
    """
    Write a Pandas DataFrame to a Delta table.

    Parameters:
    df (pd.DataFrame): The DataFrame to be written.
    tblname (str): The name of the Delta table.

    Returns:
    None
    """
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.to_table(name=tblname, format='delta')

def writeTrainTest(writer_fn, X_train, X_test, y_train, y_test, catalog, schema, rootTableName, country):
    """
    Write training and testing data to Delta tables.

    Parameters:
    writer_fn (function): The function to write DataFrames to Delta tables.
    X_train (pd.DataFrame): Training feature set.
    X_test (pd.DataFrame): Testing feature set.
    y_train (pd.Series): Training labels.
    y_test (pd.Series): Testing labels.
    catalog (str): The catalog name.
    schema (str): The schema name.
    rootTableName (str): The root table name.
    country (str): Country identifier for the table names.

    Returns:
    None
    """
    namespace = f"`{catalog}`.{schema}.{rootTableName}"

    import pyspark.pandas as ps
    import pandas as pd
    X_train_spark = ps.from_pandas(X_train.reset_index())
    y_train_spark = ps.from_pandas(pd.DataFrame(y_train).reset_index())
    X_test_spark = ps.from_pandas(X_test.reset_index())
    y_test_spark = ps.from_pandas(pd.DataFrame(y_test).reset_index())

    writer_fn(X_train_spark, f"{namespace}_X_train_{country}")
    writer_fn(y_train_spark, f"{namespace}_y_train_{country}")
    writer_fn(X_test_spark, f"{namespace}_X_test_{country}")
    writer_fn(y_test_spark, f"{namespace}_y_test_{country}")

def variance_threshold_selector(data, threshold=0.5):
    """
    Select features based on variance threshold.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    threshold (float): The variance threshold.

    Returns:
    pd.DataFrame: The DataFrame with selected features.
    """
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

def get_config_paths(pattern, config, notebook_path):
    """
    Get the YAML root path and the country-specific config path.

    Parameters:
    pattern (str): A string pattern to locate the root path in the notebook path.
    config (str): The path to the country-specific configuration file.
    notebook_path (str): The path of the current notebook.

    Returns:
    tuple: A tuple containing the YAML root path and the country-specific config path.
    """
    yaml_root = notebook_path[:notebook_path.index(pattern)]
    config_path = f"""{yaml_root}{config}"""
    print(f"config path is {config_path}")
    print(f"root path is {yaml_root}")
    return (yaml_root, config_path)

def yml_loader(yaml_root):
    """
    Load and parse a YAML configuration file.

    Parameters:
    yaml_root (str): The path to the YAML configuration file.

    Returns:
    dict: A dictionary containing the parsed YAML configuration.
    """
    import yaml
    with open(yaml_root, 'r') as file:
        config_values = yaml.safe_load(file)
    return config_values
