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



def compute_features_repurchase_fn(config_values, input_catalog, country, pk,spark):
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import pyspark.pandas as ps
    from pyspark.sql.functions import monotonically_increasing_id
    """
    Compute features related to customer repurchase behavior.

    Parameters:
    - config_values (dict): Configuration values containing schema name, table names, start date, repurchase window, and quantile ranges.
    - input_catalog (str): The input catalog name.
    - country (str): The country code to filter the data.
    - pk (str): The primary key column name.

    Returns:
    - DataFrame: A Spark DataFrame with computed features for customer repurchase behavior.

    The function performs the following steps:
    1. Extracts configuration values for schema, table names, start date, repurchase window, and quantile ranges.
    2. Constructs and executes a SQL query to create a dataset with various customer attributes and repurchase-related features.
    3. Computes additional attributes for the Customer Lifetime Value (CLV) model, such as repurchase window status, repurchase flag, first purchase status, and lifetime value.
    4. Drops columns with all missing values and adds a monotonically increasing ID as the primary key.
    5. Returns the final Spark DataFrame with the computed features.
    """
    # Extract configuration values for schema, table names, start date, repurchase window, and quantile ranges

    schema_name = config_values['schema_name']
    sales_table_name = config_values['table_types']['sales']['table_value']
    appointments_table_name = config_values['table_types']['appointments']['table_value']
    contacts_table_name = config_values['table_types']['contacts']['table_value']
    start_date = str(config_values['start_date'])
    repurchase_window = config_values['repurchase_window']

    quantile_range_0_lower = config_values['quantiles'][0]['lower_limit']
    quantile_range_0_upper = config_values['quantiles'][0]['upper_limit']

    quantile_range_1_lower = config_values['quantiles'][1]['lower_limit']
    quantile_range_1_upper = config_values['quantiles'][1]['upper_limit']

    quantile_range_2_lower = config_values['quantiles'][2]['lower_limit']
    quantile_range_2_upper = config_values['quantiles'][2]['upper_limit']

    quantile_range_3_lower = config_values['quantiles'][3]['lower_limit']
    quantile_range_3_upper = config_values['quantiles'][3]['upper_limit']



    print('Metadata Read!')

    # Create dataset
    customer_att_query = f"""
    WITH sales AS (
            SELECT ContactCode, InvoiceDate, NetAmountExclVAT, SerialNo, IsHI, IsSold, ReturnReasonCode, OpportunityCode
            FROM {input_catalog}.{schema_name}.{sales_table_name}
            WHERE CountryCode = '{country}'
            AND InvoiceDate >= '{start_date}'
        )
        
        , appointments AS (
            SELECT CAST(Date AS DATE) AS Date, ContactCode, Attended, Duration, NoShow
            FROM {input_catalog}.{schema_name}.{appointments_table_name}
            WHERE CountryCode = '{country}'
            AND CAST(Date AS DATE) >= '{start_date}'
        )

        , customers_features_base AS (
            SELECT 
                ContactCode
                , CountryCode
                , ENTRefferal
                , NoMarketingMail
                , NoMarketingPhone
                , NoMarketingSMS
                , IsOnlineContact
                , CombinedHearingLoss
                , Gender
                , TerritoryCode
                , HasQualifiedHL
        FROM {input_catalog}.{schema_name}.{contacts_table_name}
        WHERE CountryCode = '{country}'
        )

        , first_hi_purchase AS (
            SELECT ContactCode, MIN(InvoiceDate) AS first_hi_purchase_date
            FROM sales
            WHERE IsHI=1
            GROUP BY ContactCode
        )

        , repurchase_dates AS (
            SELECT *, 
                    DATEADD(MONTH, 6, first_hi_purchase_date) AS repurchase_start_date,
                    DATEADD(MONTH, {repurchase_window*12}, first_hi_purchase_date) AS repurchase_end_date     
            FROM first_hi_purchase
        )

        , pre_repurchase_appointments AS (
            SELECT a.ContactCode, SUM(a.Duration) AS pre_repurchase_app_duration, SUM(CAST(a.Attended AS int)) AS pre_repurchase_app_attended, SUM(CAST(a.NoShow AS int)) AS pre_repurchase_app_noshow
            FROM appointments a 
            INNER JOIN repurchase_dates rd ON a.ContactCode = rd.ContactCode AND a.Date <= rd.repurchase_start_date
            GROUP BY a.ContactCode
        )

        , subsequent_hi_sales AS (
            SELECT s2.ContactCode, CAST(InvoiceDate AS DATE) AS invoice_date
            FROM sales AS s2
            INNER JOIN repurchase_dates rd2 ON s2.ContactCode = rd2.ContactCode AND s2.InvoiceDate >= rd2.repurchase_start_date AND s2.InvoiceDate <= rd2.repurchase_end_date
            WHERE IsSold=1
            AND IsHI=1    -- Here I'm only interested in HI repurchases for the classifier, so I need this filter
            AND NetAmountExclVAT > 0
            AND ReturnReasonCode IS NULL
            GROUP BY s2.ContactCode, CAST(InvoiceDate AS DATE)
        )

        , subsequent_all_sales_value AS (
            SELECT s3.ContactCode, SUM(NetAmountExclVAT) AS value
            FROM sales AS s3
            INNER JOIN repurchase_dates rd3 ON s3.ContactCode = rd3.ContactCode AND s3.InvoiceDate >= rd3.repurchase_start_date AND s3.InvoiceDate <= rd3.repurchase_end_date
            WHERE IsSold=1
            AND NetAmountExclVAT > 0
            AND ReturnReasonCode IS NULL
            GROUP BY s3.ContactCode, CAST(InvoiceDate AS DATE)
        )

        , subsequent_sales_agg AS (
            SELECT sasv.ContactCode, COUNT(DISTINCT invoice_date) AS hi_repurchases, SUM(value) AS lifetime_value
            FROM subsequent_hi_sales shs
            RIGHT JOIN subsequent_all_sales_value sasv ON shs.ContactCode = sasv.ContactCode
            GROUP BY sasv.ContactCode
        )

        SELECT cfb.*
                , rd3.first_hi_purchase_date
                , rd3.repurchase_start_date
                , rd3.repurchase_end_date
                , pra.pre_repurchase_app_duration
                , pra.pre_repurchase_app_attended
                , pra.pre_repurchase_app_noshow
                , ssa.hi_repurchases
                , ssa.lifetime_value
        FROM customers_features_base cfb
        LEFT JOIN repurchase_dates rd3 ON cfb.ContactCode = rd3.ContactCode
        LEFT JOIN pre_repurchase_appointments pra ON cfb.ContactCode = pra.ContactCode
        LEFT JOIN subsequent_sales_agg ssa ON cfb.ContactCode = ssa.ContactCode
        ;
        """

    customer_att = spark.sql(customer_att_query).pandas_api()  

    # Attributes for CLV model
    # Has repurchase window passed
    customer_att['repurchase_window_passed'] = 0
    customer_att.loc[customer_att['repurchase_end_date'] < datetime.today().strftime('%Y-%m-%d') ,'repurchase_window_passed'] = 1
    customer_att['repurchase_flag'] = customer_att['hi_repurchases'].apply(lambda x: 1 if x>=1 else 0)
    customer_att['first_purchase_made'] = customer_att['first_hi_purchase_date'].apply(lambda x: 1 if not pd.isna(x) else 0)
    customer_att['lifetime_value'] = customer_att['lifetime_value'].fillna(-1).astype(int).replace(-1, None)
    customer_att['HasQualifiedHL'] = customer_att["HasQualifiedHL"].apply(lambda x: x.replace(" ","_"))
    customer_att = customer_att.dropna(axis=1, how='all')
    customer_att=customer_att.to_spark().withColumn(pk, monotonically_increasing_id())
    return customer_att