def lifetimevalue_preprocess_data(feature_df, config_values,env, country_config_values):
    """
    Preprocess the feature DataFrame for lifetime value prediction.

    Parameters:
    feature_df (DataFrame): Spark DataFrame containing the features.
    config_values (dict): Configuration values including feature names and thresholds.
    country_config_values (dict): Country-specific configuration values including quantiles.

    Returns:
    tuple: A tuple containing the training and testing datasets (X_train, X_test, y_train, y_test).
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    from utils import variance_threshold_selector

    # Convert Spark DataFrame to Pandas DataFrame
    input_df = feature_df.toPandas()

    input_df = input_df[input_df['repurchase_window_passed'] == 1].fillna(0)

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

    quantile_mapping = pd.DataFrame([
        {'quantile': 0, 'min': country_config_values['quantiles'][0]['lower_limit'], 'max': country_config_values['quantiles'][0]['upper_limit']},
        {'quantile': 1, 'min': country_config_values['quantiles'][1]['lower_limit'], 'max': country_config_values['quantiles'][1]['upper_limit']},
        {'quantile': 2, 'min': country_config_values['quantiles'][2]['lower_limit'], 'max': country_config_values['quantiles'][2]['upper_limit']},
        {'quantile': 3, 'min': country_config_values['quantiles'][3]['lower_limit'], 'max': country_config_values['quantiles'][3]['upper_limit']}
    ])


    #lambda to apply the quantile class based on the range
    map_quantile_to_value = lambda value: next((row['quantile'] for index, row in categories.iterrows() if row['min'] <= value <= row['max']), 4) 
    #4 is unknown class , XGB need numeric class values

    # Applying the function to the sales dataframe
    categories = quantile_mapping
    input_df['value_bucket'] = input_df['lifetime_value'].apply(map_quantile_to_value)
    y = input_df['value_bucket']

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
    if env != "prod" :
        X_train = X_train.sample(frac =.25)
        X_test = X_test.sample(frac =.25)
        y_train = y_train.sample(frac =.25)
        y_test = y_test.sample(frac =.25)

    return X_train, X_test, y_train, y_test

def repurchase_preprocess_data(feature_df, config_values,env,country_config_values=None):
    """
    Preprocess the input feature DataFrame based on the provided configuration values.

    Parameters:
    feature_df (DataFrame): Spark DataFrame containing the features.
    config_values (dict): Configuration dictionary containing feature names and other parameters.

    Returns:
    tuple: A tuple containing the processed training and test sets (X_train, X_test, y_train, y_test).
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    from utils import variance_threshold_selector

    # Convert Spark DataFrame to Pandas DataFrame
    input_df = feature_df.toPandas()

    # Filter rows where 'repurchase_window_passed' is 1 and fill NaN values with 0
    input_df = input_df[input_df['repurchase_window_passed'] == 1].fillna(0)

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

    # Define the target variable
    y = input_df['repurchase_flag']

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config_values['test_size'])
    if env != "prod" :
        X_train = X_train.sample(frac =.25)
        X_test = X_test.sample(frac =.25)
        y_train = y_train.sample(frac =.25)
        y_test = y_test.sample(frac =.25)

    return (X_train, X_test, y_train, y_test)