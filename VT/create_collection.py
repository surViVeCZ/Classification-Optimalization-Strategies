import pymongo
import logging
from tqdm import tqdm

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB connection details
mongo_uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(mongo_uri)
db = client['drdb']

def create_collection_with_verified_domains(verified_domains_file, source_collection_name, new_collection_name):
    with open(verified_domains_file, 'r') as file:
        verified_domains = set(line.strip() for line in file)
    
    logging.info(f"Found {len(verified_domains)} verified domains.")
    
    source_collection = db[source_collection_name]
    new_collection = db[new_collection_name]

    logging.info(f"Creating new collection {new_collection_name} with verified domains.")
    for domain in tqdm(verified_domains, desc=f"Adding documents to {new_collection_name}"):
        record = source_collection.find_one({'domain_name': domain})
        if record:
            new_collection.insert_one(record)
        else:
            logging.warning(f"Document with domain {domain} not found in {source_collection_name}.")
    logging.info(f"Created new collection {new_collection_name} with verified domains.")
    

def add_collection_without_duplicates(source_collection_name, target_collection_name):
    source_collection = db[source_collection_name]
    target_collection = db[target_collection_name]
    
    logging.info(f"Adding documents from {source_collection_name} to {target_collection_name} without duplicates.")
    
    existing_ids = set(target_collection.find().distinct('_id'))
    bulk_ops = []
    for record in tqdm(source_collection.find({'_id': {'$nin': list(existing_ids)}}), desc=f"Adding documents to {target_collection_name}"):
        bulk_ops.append(pymongo.InsertOne(record))

    if bulk_ops:
        target_collection.bulk_write(bulk_ops)
    
    logging.info(f"Added {len(bulk_ops)} documents to {target_collection_name}.")

def main():
    # verified_domains_file = 'ondra_domains_verified.txt'
    # phishing_since_2402_collection_name = 'phishing_since_2402'
    # misp_2311_collection_name = 'misp_2311'
    # new_collection_name = 'misp_2402'
    
    # logging.info("Starting process...")
    # create_collection_with_verified_domains(verified_domains_file, phishing_since_2402_collection_name, new_collection_name)
    # add_collection_without_duplicates(misp_2311_collection_name, new_collection_name)
    
    # logging.info("Process completed.")

    #add benign_2310 to new collection benign_2402
    benign_2310_collection_name = 'benign_2310'
    benign_2402_collection_name = 'benign_2402'
    add_collection_without_duplicates(benign_2310_collection_name, benign_2402_collection_name)

if __name__ == '__main__':
    main()
