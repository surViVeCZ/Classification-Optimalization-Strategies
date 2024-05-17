from pandas import DataFrame, Series, concat
from ._helpers import map_dict_to_series


def map_ip_data(ip_data):
    if ip_data is None:
        return None, None, None
    else:
        ip_data = [ip for ip in ip_data if ip['geo'] is not None]
        return [ip['geo']['country'] for ip in ip_data], [ip['geo']['latitude'] for ip in ip_data], [
            ip['geo']['longitude'] for ip in ip_data]

def map_experimental_ip_data(ip_data):
    if ip_data is None:
        return None, None, None
    else:
        ip_data = [ip for ip in ip_data if ip['geo'] is not None]
        return [ip['geo']['isp'] for ip in ip_data], [ip['geo']['org'] for ip in ip_data], [
            ip['geo']['region'] for ip in ip_data]

# This is just a function that takes nested fields and surfaces them
# into their own columns. Sounds simple, but it's not. The fields
# are nested in a variety of ways, and some of them are lists of
# dictionaries.
#
# This is probably a pretty slow way to do it, but it works.
# If you are a pandas wizard and know how to do this faster, 
# please make it so. This is the slowest transformation by far.
def flatten_geo(df: DataFrame) -> DataFrame:
    """
    Surface select nested fields in a dataframe into own columns.
    Input: DF with nested fields
    Output: DF with new columns for the fields
    """
    # flatten ip_data
    df["countries"], df["latitudes"], df["longitudes"] = zip(*df["ip_data"].apply(map_ip_data))

    # EXPERIMENTAL: additional ip_data (currently not used)
    # df["isps"], df["orgs"], df["regions"] = zip(*df["ip_data"].apply(map_experimental_ip_data))

    return df
