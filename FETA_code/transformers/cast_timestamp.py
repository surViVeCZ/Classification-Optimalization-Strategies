import numpy as np
from pandas import DataFrame, api, Int64Dtype
from pandas.core.dtypes import common as com

def cast_timestamp(df: DataFrame):
    """
    Cast timestamp fields to seconds since epoch.
    """
    # Original version:
    # for col in df.columns:
    #   if api.types.is_datetime64_ns_dtype(df[col]) or api.types.is_timedelta64_ns_dtype(df[col]):
    #       df[col] = df[col].dt.floor('s').astype(np.int64)
    #       # replace minimum int with null
    #    df[col] = df[col].replace(np.iinfo(np.int64).min, np.nan)

    for col in df.columns:
        if com.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()  # This converts timedelta to float (seconds)
        elif com.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(np.int64) // 10**9  # Converts datetime64 to Unix timestamp (seconds)

    return df