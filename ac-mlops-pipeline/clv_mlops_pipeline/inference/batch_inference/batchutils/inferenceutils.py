def lifetimevalue_preprocess_inferencedata(inference_df, config_values,primary_key_columns,spark):
    """
    Preprocess the feature DataFrame for lifetime value prediction.

    Parameters:
    feature_df (DataFrame): Spark DataFrame containing the features.
    config_values (dict): Configuration values including feature names and thresholds.
    country_config_values (dict): Country-specific configuration values including quantiles.

    Returns :
    tuple: A tuple containing the training and testing datasets (X_train, X_test, y_train, y_test).
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    from utils import variance_threshold_selector

    # Convert Spark DataFrame to Pandas DataFrame
    input_df = inference_df.toPandas()

    input_df = input_df.fillna(0)

    keys_df = input_df[primary_key_columns]


    # Select numerical features
    X = input_df[config_values['features']['numerical']]

    # Create dummy variables for categorical features
    dummies = pd.DataFrame()
    for colname in config_values['features']['categorical']:
        tempdf = pd.get_dummies(input_df[colname], prefix=colname, drop_first=True)
        dummies = pd.concat([dummies, tempdf], axis=1)

    # Merge the dummies with other features
    X = pd.concat([X, dummies], axis=1)

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit(X).transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Apply variance threshold selector
    min_variance = config_values['min_variance']
    low_variance = variance_threshold_selector(X, min_variance)
    X = low_variance

    # Merge the primary keys with the processed features
    final_df = pd.concat([keys_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Convert back to Spark DataFrame
    spark_df = spark.createDataFrame(final_df)

    return spark_df





def repurchase_preprocess_inference_data(inference_df, config_values,primary_key_columns,spark):
    from sklearn.preprocessing import MinMaxScaler
    from utils import variance_threshold_selector
    import pandas as pd

    # Convert Spark DataFrame to Pandas DataFrame
    input_df = inference_df.toPandas()

    # Filter rows where 'repurchase_window_passed' is 1 and fill NaN values with 0
    input_df = input_df.fillna(0)

   
    keys_df = input_df[primary_key_columns]

    # Select numerical features
    X = input_df[config_values['features']['numerical']]

    # Create dummy variables for categorical features
    dummies = pd.DataFrame()
    for colname in config_values['features']['categorical']:
        tempdf = pd.get_dummies(input_df[colname], prefix=colname, drop_first=True)
        dummies = pd.concat([dummies, tempdf], axis=1)

    # Merge the dummies with numerical features
    X = pd.concat([X, dummies], axis=1)

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit(X).transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # Apply variance threshold selector
    min_variance = config_values['min_variance']
    low_variance = variance_threshold_selector(X, min_variance)
    X = low_variance

    # Merge the primary keys with the processed features
    final_df = pd.concat([keys_df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    # Convert back to Spark DataFrame
    spark_df = spark.createDataFrame(final_df)


    return spark_df