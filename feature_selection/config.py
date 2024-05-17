from os import getenv
from dotenv import load_dotenv

load_dotenv()


class Config:
    # MongoDB setup
    MONGO_URI = getenv("DR_MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB = "drdb"
    # Collections to process by loader - {label: collection_name}
    COLLECTIONS = {
        # "phishing_2307": "misp_2307",
        "benign_2310": "benign_2310",
        # "misp_2310": "misp_2310",
        # "cesnet3_2311":"cesnet3_2311",
        # "phishing_since_2402":"phishing_since_2402"
    }
