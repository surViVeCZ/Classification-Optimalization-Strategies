from pandas import DataFrame

def label(df: DataFrame) -> DataFrame:
    """
    Combine label columns into one.
    Input: DF with label, category
    Output: DF where label column is both combined and category is dropped
    """
    df['label'] = df['label'] + ':' + df['category']
    df.drop(columns=['category'], inplace=True)
    return df
