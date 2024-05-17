# transform functions for dataframes
# [DF with projected fields] -> transformer 1 -> transformer 2 -> ... -> [DF for training]
# executed in order of appearance in this file

# transform functions must take a pandas DataFrame as input and return a DataFrame as output
# IMPORTANT! import them here as transform_<name> so they can be called automatically in loader.py
#                             ===================

# to save an intermediate dataframe after a transform, append _save to the imported name

# combine label and category into a single field
from .label import label as transform_label
# transform DNS
from .dns import dns as transform_dns_save
# calculate IP-related features
from .ip import ip as transform_ip_save
# flatten nested IP Geo fields
from .flatten_geo import flatten_geo as transform_flatten
# analyze TLS certificates
from .tls import tls as transform_tls_save
# calculate length of domain name
from .lexical import lex as transform_lexical_save
# calculate standard deviation of latitudes and longitudes
from .geo import geo as transform_geo
# transform RDAP
from .rdap import rdap as transform_rdap

# drop non-training columns (done before training)
# from .drop_nontrain import drop_nontrain as transform_drop
