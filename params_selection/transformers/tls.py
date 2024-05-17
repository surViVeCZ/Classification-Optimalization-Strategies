import datetime
import re
from pandas import DataFrame, Series, concat
from pandas.errors import OutOfBoundsDatetime

def run_analyzer(row: Series) -> Series:
    """
    Run all analyzers on the row.
    """
    tls = row["tls"]
    date = row["tls_evaluated_on"]
    analysis_result = analyze_tls(tls, date)
    return Series(analysis_result['features'])

def tls(df: DataFrame) -> DataFrame:
    """
    Transform tls field into new columns.
    Input: DF with tls field
    Output: DF with new columns for the fields
    """
    tls_columns = df.apply(run_analyzer, axis=1)
    df.drop(columns=['tls'], inplace=True)
    df = concat([df, tls_columns], axis=1)
    return df

def encodePolicy(oid):
    if not oid:
        return 0
    encoded = int(oid.replace('.',''))
    return encoded % 2147483647

tls_version_ids = {
    "TLSv1.0": 0,
    "TLSv1.1": 1,
    "TLSv1.2": 2,
    "TLSv1.3": 3
}

tls_cipher_ids = {
    'ECDHE-RSA-AES128-GCM-SHA256': 0,
    'TLS_AES_128_GCM_SHA256': 1,
    'TLS_AES_256_GCM_SHA384': 2,
    'ECDHE-ECDSA-CHACHA20-POLY1305': 3,
    'ECDHE-RSA-AES128-SHA': 3,
    'ECDHE-RSA-CHACHA20-POLY1305': 4,
    'ECDHE-RSA-AES256-GCM-SHA384': 5,
    'ECDHE-ECDSA-AES128-GCM-SHA256': 6,
    'TLS_CHACHA20_POLY1305_SHA256': 7,
    'ECDHE-ECDSA-AES256-GCM-SHA384': 8,
    'ECDHE-RSA-AES256-SHA384': 9,
    'AES128-SHA256': 10,
    'DHE-RSA-AES256-GCM-SHA384': 11,
    'AES256-SHA256': 12,
    'AES128-SHA': 13,
    'DHE-RSA-AES128-GCM-SHA256': 14
}

"""   
@param item: one tls field from database
@param collection_date: date when the collection was made
@return: return {"success": True/False, "features": dict/None}
"""  
def analyze_tls(item: dict, collection_date: datetime.datetime) -> dict:
    # We dont hane tls data for this domain
    if item is None:
        features = { 
            "tls_has_tls": False,                           # Has TLS
            "tls_chain_len": None,                          # Length of certificate chain
            "tls_negotiated_version_id": None,              # Evaluated TLS version
            "tls_negotiated_cipher_id": None,               # Evaluated cipher
            "tls_root_cert_validity_len": None,             # Total validity time of root certificate
            "tls_root_cert_lifetime": None,                 # How long was the root certificate valid at the time of collection
            #NOTUSED# "tls_root_cert_validity_remaining": None, # Time to expire of root certificate from time of collection
            "tls_leaf_cert_validity_len": None,             # Total validity time of leaf certificate      
            "tls_leaf_cert_lifetime": None,                 # How long was the leaf certificate valid at the time of collection
            #NOTUSED# "tls_leaf_cert_validity_remaining": None,  # Time to expire of leaf certificate from time of collection      
            #NOTUSED# "tls_mean_certs_validity_len": None,       # Mean validity time of all certificates in chain including root
            "tls_broken_chain": None,                       # Chain was never valid, 
            "tls_expired_chain": None,                      # Chain already expired at time of collection
            "tls_total_extension_count": None,              # Total number of extensions in certificate
            "tls_critical_extensions": None,                # Total number of critical extensions in certificate
            "tls_with_policies_crt_count": None,            # Number of certificates enforcing specific encryption policy
            "tls_percentage_crt_with_policies": None,       # Percentage of certificates enforcing specific encryption policy
            "tls_x509_anypolicy_crt_count": None,           # Number of certificates enforcing X509 - ANY policy
            "tls_iso_policy_crt_count": None,               # Number of certificates supporting Joint ISO-ITU-T policy (OID root is 1)
            "tls_joint_isoitu_policy_crt_count": None,      # Number of certificates supporting Joint ISO-ITU-T policy (OID root is 2)
            #NOTUSED# "tls_iso_policy_oid": None,           # OID of ISO policy (if any or 0)
            #NOTUSED# "tls_isoitu_policy_oid": None,        # OID of ISOITU policy (if any or 0)
            #NOTUSED# "tls_unknown_policy_crt_count": None, # How many cerificates uses unknown (not X509v3, not version 1, not version 2) policy
            "tls_subject_count": None,                      # How many subjects can be found in SAN extension (can be linked to phishing)       
            "tls_server_auth_crt_count": None,              # How many certificates are used for server authentication (can be simultanously used for client authentication)      
            "tls_client_auth_crt_count": None,              # How many certificates are used for client authentication
            #NOTUSED# "CA_certs_in_chain_count": None,      # Count of certificates that are also CA (can sign other certificates)
            "tls_CA_certs_in_chain_ratio": None,            # Ration of CA certificates in chain
            "tls_unique_SLD_count": None,                   # Number of unique SLDs in SAN extension
            #NOTUSED# "tls_common_names": None,             # List of common names in certificate chain (CATEGORICAL!)
            "tls_common_name_count": None,                  # Number of common names in certificate chain
    }
    
        return {"success": False, "features": features}
    
    SSL_SCORE = 0
        
    # FEATURES
    tls_version_id = tls_version_ids.get(item['protocol'] , 0)
    tls_cipher_id = tls_cipher_ids.get(item['cipher'] , 0)
    
    # Certificate features #
    common_names = []
    
    ### ROOT CERTIFICATE ###
    root_crt_validity_len = -1
    root_crt_lifetime = -1
    #root_crt_time_to_expire = -1
    
    
    ### LEAF CERTIFICATE ###
    leaf_crt_validity_len = -1
    leaf_crt_lifetime = -1
    #leaf_cert_time_to_live = -1
    
    
    ### BAIC FEATURES ###
    mean_cert_len = -1
    broken_chain = 0
    expired_chain = 0
    
    
    ### EXTENSION FEATURES ###
    total_extension_count = -1
    critical_extensions = -1
    any_policy_cnt = 0
    percentage_of_policies = 0
    server_auth = 0
    client_auth = 0
    unknown_policy_cnt = 0
    X_509_used_cnt = 0
    iso_policy_used_cnt = 0
    isoitu_policy_used_cnt = 0
    iso_policy_oid = None
    isoitu_policy_oid = None
    CA_count = 0  # Ration of CA certificates in chain
    CA_ratio = 0
    
    ### NUMBER OF SUBJECTS if SAN ###
    subject_count = 0

    ### NUMBER OF SLDs
    SLD_cnt = 0

    #### Procesing of certificates and root especialy ####
    if len(item['certificates']) == 0:
        return {"success": False, "error": "No certificates"}
    mean_len = 0
    cert_counter = 0
    for certificate in item['certificates']: 
        cert_counter += 1
        
        validity_len = round(int(certificate['valid_len']) / (60*60*24))
        
        if validity_len < 0:
            broken_chain = 1
            break
        
        try:
            lifetime = round((collection_date - certificate['validity_start']).total_seconds() / (60*60*24))
        except OutOfBoundsDatetime:
            print(certificate['validity_start'], collection_date)
            lifetime = -1

        try:
            time_to_expire = round((certificate['validity_end'] - collection_date).total_seconds() / (60*60*24))
        except OutOfBoundsDatetime:
            print(certificate['validity_end'], collection_date)
            time_to_expire = -1
        
        if time_to_expire < 0:
            expired_chain = 1
            break
    
        if cert_counter == 1:
            leaf_crt_validity_len = validity_len
            leaf_crt_lifetime = lifetime
            #leaf_cert_time_to_live = time_to_expire
            
            
        if certificate['is_root']:
            root_crt_validity_len = validity_len
            root_crt_lifetime = lifetime
            #root_crt_time_to_expire = time_to_expire
        
        if certificate['common_name']:
            common_names.append(certificate['common_name'])
        
            # if the certificate is not valid now it is suspicious
        mean_len += validity_len  

        mean_cert_len = mean_len / cert_counter
        
        
        #### EXTENSIONS ####
        total_extension_count += len(certificate['extensions'])
        for extension in certificate['extensions']:
            if extension['critical']:
                critical_extensions += 1
                
                
            if extension['name'] == "subjectAltName" and extension['value'] is not None:
                subject_count = len(extension['value'].split(","))

                # count SLDs
                unique_SLDs = set()
                for name in extension['value'].split(","):
                    if "DNS:" in name:
                        sld = name.split("DNS:")[1].split(".")
                        if len(sld) >= 2:
                            unique_SLDs.add(sld[-2])
                SLD_cnt = len(unique_SLDs)
            
                            
            if extension["name"] == "extendedKeyUsage" and extension["value"] is not None:
                # apend extension [value] to file issuers.txt
                auth_type = extension["value"].split(", ")
                
                for auth in auth_type:
                    if auth == "TLS Web Server Authentication":
                        server_auth += 1
                    if auth == "TLS Web Client Authentication":
                        client_auth += 1    
                        
            if extension["name"] == "certificatePolicies" and extension["value"] is not None:
                any_policy_cnt += 1
                data = extension["value"].split(",")
                
                # for each value remove everythong after newline 
                data = [x.split("\n")[0] for x in data]
                # filter values starting only with Policy:
                data = list(filter(lambda x: x.startswith("Policy:"), data))
                
                
                for policy in data:
                            
                    # match regex to policy
                    if re.compile(r"(.)*X509v3(.)*").match(policy):
                        X_509_used_cnt += 1
                    elif re.compile(r"Policy: 1\.").match(policy):
                        iso_policy_used_cnt += 1
                        iso_policy_oid = re.search('1\.[0-9\.]+', policy).group()
                    elif re.compile(r"Policy: 2\.").match(policy):
                        isoitu_policy_used_cnt += 1
                        isoitu_policy_oid = re.search('2\.[0-9\.]+', policy).group()
                    else:
                        unknown_policy_cnt += 1
            
            if extension["name"] == "basicConstraints":
                if extension["value"] == "CA:TRUE":
                    CA_count += 1
                
    # computation of certificate chain fetures
    percentage_of_policies  = (any_policy_cnt / cert_counter)
    CA_ratio = (CA_count / cert_counter)
    
    
    ### roud float valuet to 1 decimal place ###
    mean_cert_len = round(mean_cert_len, 1)
    #CA_ratio = round(CA_ratio, 1)
    #percentage_of_policies = round(percentage_of_policies, 1)
    
    # Return dictionary with all features
    features = { 
        "tls_has_tls": True,                                         # Has TLS
        "tls_chain_len": item['count'],                              # Length of certificate chain
        "tls_negotiated_version_id": tls_version_id,                 # Evaluated TLS version
        "tls_negotiated_cipher_id": tls_cipher_id,                   # Evaluated cipher
        "tls_root_cert_validity_len": root_crt_validity_len,         # Total validity time of root certificate
        "tls_root_cert_lifetime": root_crt_lifetime,                 # How long was the root certificate valid at the time of collection
        #NOTUSED# "tls_root_cert_validity_remaining": root_crt_time_to_expire, # Time to expire of root certificate from time of collection
        "tls_leaf_cert_validity_len": leaf_crt_validity_len,         # Total validity time of leaf certificate      
        "tls_leaf_cert_lifetime": leaf_crt_lifetime,                 # How long was the leaf certificate valid at the time of collection
        #NOTUSED# "tls_leaf_cert_validity_remaining": leaf_cert_time_to_live,  # Time to expire of leaf certificate from time of collection      
        #NOTUSED# "tls_mean_certs_validity_len": mean_cert_len,      # Mean validity time of all certificates in chain including root
        "tls_broken_chain": broken_chain,                            # Chain was never valid, 
        "tls_expired_chain": expired_chain,                          # Chain already expired at time of collection
        "tls_total_extension_count": total_extension_count,          # Total number of extensions in certificate
        "tls_critical_extensions": critical_extensions,              # Total number of critical extensions in certificate
        "tls_with_policies_crt_count": any_policy_cnt,               # Number of certificates enforcing specific encryption policy
        "tls_percentage_crt_with_policies": percentage_of_policies,  # Percentage of certificates enforcing specific encryption policy
        "tls_x509_anypolicy_crt_count": X_509_used_cnt,              # Number of certificates enforcing X509 - ANY policy
        "tls_iso_policy_crt_count": iso_policy_used_cnt,             # Number of certificates supporting Joint ISO-ITU-T policy (OID root is 1)
        "tls_joint_isoitu_policy_crt_count": isoitu_policy_used_cnt, # Number of certificates supporting Joint ISO-ITU-T policy (OID root is 2)
        #NOTUSED# "tls_iso_policy_oid": encodePolicy(iso_policy_oid),          # OID of ISO policy (if any or 0)
        #NOTUSED# "tls_isoitu_policy_oid": encodePolicy(isoitu_policy_oid),    # OID of ISOITU policy (if any or 0)
        #NOTUSED# "tls_unknown_policy_crt_count": unknown_policy_cnt,# How many cerificates uses unknown (not X509v3, not version 1, not version 2) policy
        "tls_subject_count": subject_count,                          # How many subjects can be found in SAN extension (can be linked to phishing)       
        "tls_server_auth_crt_count": server_auth,                    # How many certificates are used for server authentication (can be simultanously used for client authentication)      
        "tls_client_auth_crt_count": client_auth,                    # How many certificates are used for client authentication
        #NOTUSED# "CA_certs_in_chain_count": CA_count,               # Count of certificates that are also CA (can sign other certificates)
        "tls_CA_certs_in_chain_ratio": CA_ratio,                     # Ration of CA certificates in chain
        "tls_unique_SLD_count": SLD_cnt,                             # Number of unique SLDs in SAN extension
        #NOTUSED# "tls_common_names": common_names,                  # List of common names in certificate chain (CATEGORICAL!)
        "tls_common_name_count": len(common_names),                  # Number of common names in certificate chain
    }
    
    return {"success": True, "features": features}