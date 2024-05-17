from typing import Optional, List, Dict
import numpy as np
from pandas import DataFrame, Series
from ._helpers import get_normalized_entropy
from .lexical import count_subdomains
import schema


def dns(df: DataFrame) -> DataFrame:
    df = add_dns_record_counts(df)

    # Domain name level
    #NOTUSED# df["dns_dn_level"] = df["domain_name"].apply(lambda dn: count_subdomains(dn))

    # Zone DN info
    df["dns_zone"].fillna('', inplace=True)
    df["dns_zone_level"] = df["dns_zone"].apply(lambda dn: count_subdomains(dn))
    df["dns_zone_digit_count"] = df["dns_zone"].str.count(r'\d')
    df["dns_zone_len"] = df["dns_zone"].str.len()
    df["dns_zone_entropy"] = df["dns_zone"].apply(lambda dn: get_normalized_entropy(dn))

    # Count resolved record types
    df["dns_resolved_record_types"] = df.apply(count_resolved_record_types, axis=1)

    # DNSSEC features
    df["dns_has_dnskey"].replace({True: 1, False: 0, None: 0, np.nan: 0}, inplace=True)
    df["dns_dnssec_score"] = df.apply(make_dnssec_score, axis=1)

    # TTL features
    df["dns_ttl_avg"], df["dns_ttl_stdev"], df["dns_ttl_low"], df["dns_ttl_mid"], df["dns_ttl_distinct_count"] = zip(
        *df["dns_ttls"].apply(make_ttl_features))

    # SOA features
    df["dns_SOA_tmp"] = df.apply(
        lambda row: row["dns_zone_SOA"] if row["dns_SOA"] is None and row["dns_zone_level"] > 0 else row["dns_SOA"],
        axis=1)

    df["dns_soa_primary_ns_level"], df["dns_soa_primary_ns_digit_count"], df["dns_soa_primary_ns_len"], df[
        "dns_soa_primary_ns_entropy"] = zip(*df["dns_SOA_tmp"].apply(
        lambda soa: make_string_features(soa["primary_ns"]) if soa is not None else (None, None, None, None)))

    df["dns_soa_email_level"], df["dns_soa_email_digit_count"], df["dns_soa_email_len"], df[
        "dns_soa_email_entropy"] = zip(*df["dns_SOA_tmp"].apply(
        lambda soa: make_string_features(soa["resp_mailbox_dname"]) if soa is not None else (None, None, None, None)))

    #NOTUSED # There is no 0 time, could be anything
    #df["dns_soa_serial"], df["dns_soa_refresh"], df["dns_soa_retry"], df["dns_soa_expire"], df["dns_soa_min_ttl"] = zip(
    #    *df["dns_SOA_tmp"].apply(
    #        lambda soa: (soa["serial"], soa["refresh"], soa["retry"], soa["expire"], soa["min_ttl"]) if soa else (
    #            None, None, None, None, None)))
    
    df["dns_soa_refresh"], df["dns_soa_retry"], df["dns_soa_expire"], df["dns_soa_min_ttl"] = zip(
        *df["dns_SOA_tmp"].apply(
            lambda soa: (soa["refresh"], soa["retry"], soa["expire"], soa["min_ttl"]) if soa else (
                None, None, None, None)))

    df.drop(columns=["dns_SOA_tmp"], inplace=True)

    # MX features
    df["dns_domain_name_in_mx"] = df[["domain_name", "dns_MX"]].apply(
        lambda row: None if row["dns_MX"] is None else (row["domain_name"] in [x['name'] for x in row['dns_MX']]), axis=1).astype(bool)
    df["dns_mx_avg_len"], df["dns_mx_avg_entropy"] = zip(*df["dns_MX"].apply(make_mx_features))




    # TXT features
    df["dns_txt_avg_len"], df["dns_txt_avg_entropy"], df["dns_txt_external_verification_score"] = zip(*df["dns_TXT"].apply(make_txt_features))

    # E-mail/TXT features (flattening)
    df["dns_txt_spf_exists"], df["dns_txt_dkim_exists"], df["dns_txt_dmarc_exists"] = zip(
        *df["dns_email_extras"].apply(lambda e: (int(e["spf"] or 0), int(e["dkim"] or 0), int(e["dmarc"] or 0))))
    
    return df


# OK
def add_dns_record_counts(df: DataFrame) -> DataFrame:  # OK
    """
    Calculate number of DNS records for each domain.
    Input: DF with dns_* columns
    Output: DF with dns_*_count columns added
    """

    for column in [f'dns_{c}' for c in ['A', 'AAAA', 'MX', 'NS', 'TXT']]:
        df[column + '_count'] = df[column].apply(lambda values: len(values) if values is not None else 0)

    df["dns_SOA_count"] = df["dns_SOA"].apply(lambda x: 0 if x is None else 1)
    df["dns_CNAME_count"] = df["dns_CNAME"].apply(lambda x: 0 if x is None else 1)

    return df


# OK
def make_dnssec_score(row: Series) -> Optional[float]:
    has_dnskey = row["dns_has_dnskey"]
    if has_dnskey == 0:
        return 0.0

    zone_ok = row["dns_zone_dnskey_selfsign_ok"] or False
    if not zone_ok:
        return -2.0

    dnssec = row["dns_dnssec"]
    if dnssec is None:
        return 0.0

    # only consider record types that have been resolved for the dn
    values = [v for (k, v) in dnssec.items() if (row[f"dns_{k}_count"] or 0) > 0]

    if len(values) == 0:
        return 0.0

    score = 0.0
    for v in values:
        if v == 1:  # 1 = VALID_SELF_SIG
            score += 1.0
        else:  # anything else is suspicious
            score -= 2.0

    return score / len(values)


# OK
def count_resolved_record_types(row: Series) -> int:
    ret = 0
    for record_type in schema.dns_types_all:
        if row[f"dns_{record_type}"] is not None:
            ret += 1
    return ret


# OK
def make_ttl_features(ttls: Optional[Dict[str, int]]):
    if ttls is None:
        return None, None, None, None, None

    ttls = np.array([v for v in ttls.values() if v is not None])
    if len(ttls) == 0:
        return None, None, None, None, None

    mean = np.mean(ttls)
    std = np.std(ttls)

    bins = [101, 501]
    bin_counts = np.bincount(np.digitize(ttls, bins))
    total_vals = len(ttls)

    low = bin_counts[0] / total_vals if len(bin_counts) > 0 else 0
    mid = bin_counts[1] / total_vals if len(bin_counts) > 1 else 0
    dist = len(np.unique(ttls))

    return mean, std, low, mid, dist


# OK
def make_mx_features(mx: Optional[List[str]]):
    if mx is None:
        return None, None

    mx_len_sum = 0
    mx_entropy_sum = 0
    total = len(mx)

    for mailserver in mx:
        mx_len_sum += len(mailserver['name'])
        mx_entropy_sum += get_normalized_entropy(mailserver['name'])

    return mx_len_sum / total, mx_entropy_sum / total


# OK
def make_string_features(dn: str):
    return count_subdomains(dn), sum([1 for d in dn if d.isdigit()]), len(dn), get_normalized_entropy(dn)


# OK
def make_txt_features(txt: Optional[List[str]]):
    txt_entropy_sum = 0
    verification_score = 0
    verifiers = ("google-site-verification=", "ms=", "apple-domain-verification=",
                 "facebook-domain-verification=")
    total_non_empty = 0
    txt_sum_len = 0
    txt_avg_len = 0

    if txt is None:
        return None, None, 0

    for rec in txt:
        if len(rec) == 0:
            continue

        if rec is not None:
            txt_sum_len += len(rec)

        total_non_empty += 1
        entropy = get_normalized_entropy(rec)

        if entropy is not None:
            txt_entropy_sum += entropy
        else:
            txt_entropy_sum += 0

        rec = rec.lower()
        for verifier in verifiers:
            if verifier in rec:
                verification_score += 1

    txt_record_count = len(txt)
    if txt_record_count > 0:
        txt_avg_len = txt_sum_len / txt_record_count

    if txt_entropy_sum > 0:
        txt_entropy_sum = txt_entropy_sum / total_non_empty
    else:
        txt_entropy_sum = None

    return txt_avg_len, txt_entropy_sum, verification_score
