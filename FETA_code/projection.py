# This is a mongo aggregation pipeline that selects the 
# fields we want to keep for transformation.
# We use two stages:
# 1. $match to filter out documents of domains that haven't been resolved yet
# 2. $project to select the fields we want to use
#
# If you're authoring a new transformer, you'll need to add the
# fields you want to use to this pipeline.
# At the most basic level, use a 1 to select the field. 
# Fields not specified in the projection will be excluded.
# Use a custom name and add an existing (usually nested) field to it using $.
# More complex projections can be done, but try to avoid them.
# We can do complex stuff in the transformers.
#
# See the mongo docs for more info on accessing nested fields:
# https://docs.mongodb.com/manual/tutorial/project-fields-from-query-results/
#
# IMPORTANT
# After adding a new field, you'll need to add it to the schema
# in schema.py - see that file for more information.

query = {
    "evaluated_on": {"$ne": None},
    "remarks.dns_evaluated_on": {"$ne": None}
}

projection = {
    "_id": 0,
    "domain_name": 1,
    "label": 1,
    "category": 1,
    #
    "dns_dnssec": "$dns.dnssec",
    "dns_email_extras": {
        "spf": "$dns.remarks.has_spf",
        "dkim": "$dns.remarks.has_dkim",
        "dmarc": "$dns.remarks.has_dmarc",
    },
    "dns_ttls": "$dns.ttls",
    "dns_zone": "$dns.remarks.zone",
    "dns_has_dnskey": "$dns.remarks.has_dnskey",
    "dns_zone_dnskey_selfsign_ok": "$dns.remarks.zone_dnskey_selfsign_ok",
    **{f"dns_{dns_type}": f"$dns.{dns_type}" for dns_type in ['A', 'AAAA', 'SOA', 'zone_SOA', 'MX', 'TXT']},
    "dns_CNAME": "$dns.CNAME.value",
    "dns_NS": {"$map": {
        "input": {"$objectToArray": "$dns.NS"},
        "as": "item",
        "in": "$$item.k"
    }},
    "dns_MX": {"$map": {
        "input": {"$objectToArray": "$dns.MX"},
        "as": "item",
        "in": {"name": "$$item.k", "priority": "$$item.v.priority"}
    }},
    #
    "tls": 1,
    "dns_evaluated_on": "$remarks.dns_evaluated_on",
    "rdap_evaluated_on": "$remarks.rdap_evaluated_on",
    "tls_evaluated_on": "$remarks.tls_evaluated_on",
    #
    "rdap_registration_date": "$rdap.registration_date",
    "rdap_expiration_date": "$rdap.expiration_date",
    "rdap_last_changed_date": "$rdap.last_changed_date",
    # "rdap.entities.registrar.handle": 1,
    "rdap_dnssec": "$rdap.dnssec",
    "rdap_entities": "$rdap.entities", # RDAP entities for the domain
    #
    "ip_data.geo.country": 1,
    "ip_data.geo.latitude": 1,
    "ip_data.geo.longitude": 1,
    "ip_data.asn": 1,
    "ip_data.remarks.average_rtt": 1,
    "ip_data.from_record": 1,
    "ip_data.ip": 1,
    # RDAP data to each releated IP address
    "ip_data.rdap.ip_version": 1,
    "ip_data.rdap.entities": 1,
    "ip_data.rdap.network": 1,
}
