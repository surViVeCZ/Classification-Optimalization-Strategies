from pandas import DataFrame
from pyarrow import Table

import sys
from pathlib import Path
path_to_upper_root = '..'

# Add the path to sys.path if not already included
if path_to_upper_root not in sys.path:
    sys.path.append(path_to_upper_root)
from FETA_code import schema

nontraining_fields = [
    #"domain_name",
    "dns_evaluated_on",
    "rdap_evaluated_on",
    "tls_evaluated_on",

    # IP data
    "ip_data",
    "countries",
    "latitudes",
    "longitudes",

    # DNS
    "dns_dnssec",
    "dns_zone_dnskey_selfsign_ok",
    "dns_email_extras",
    "dns_ttls",
    "dns_zone",
    "dns_zone_SOA",
    *[f"dns_{t}" for t in schema.dns_types_all],

    "rdap_registration_date",
    "rdap_last_changed_date",
    "rdap_expiration_date",
    "rdap_dnssec",
    "rdap_entities"

    #"tls_root_cert_validity_remaining",
    #"tls_leaf_cert_validity_remaining"
]


def drop_nontrain_table(table: Table):
    """
    Drop non-training columns.
    """
    fields = [x for x in nontraining_fields if x in table.column_names]
    return table.drop(fields)


def drop_nontrain_df(df: DataFrame) -> DataFrame:
    """
    Drop non-training columns.
    """
    fields = [x for x in nontraining_fields if x in df.columns]
    df.drop(columns=fields, inplace=True)
    return df
