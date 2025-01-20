"""
This sample module contains  features logic that can be used to generate and populate tables in Feature Store.
You should plug in your own features computation logic in the compute_features_fn method below.
"""
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType, TimestampType
from pytz import timezone


@F.udf(returnType=StringType())
def _partition_id(dt):
    # datetime -> "YYYY-MM"
    return f"{dt.year:04d}-{dt.month:02d}"


def _filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(F.col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(F.col(ts_column) < end_date)
    return df



def compute_features(config_values, input_catalog, pk,spark):
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import pyspark.pandas as ps
    from pyspark.sql.functions import monotonically_increasing_id



    # Extract configuration values for schema, table names, start date, repurchase window, and quantile ranges

    input_schema_name = config_values['data']["input_schema"]
    input_table_name = config_values["data"]["input_table"]

    print('Metadata Read!')

    feature_df = spark.sql(f"SELECT * FROM {input_catalog}.{input_schema_name}.{input_table_name}").withColumn(pk, monotonically_increasing_id())


    return feature_df