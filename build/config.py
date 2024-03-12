# MongoDB info (username, password and port can be specified in this URL if needed)
MONGO_URL = "mongodb://127.0.0.1"

# Storage folder for wikidata dumps and other things
STORAGE_FOLDER="/wfd_build_storage"

# The Old Wikidata JSON dump date to download (set it to the model's dataset build date)
# Precision : When downloading Old wikidata, the script will look for the dump that is just after this date. 
OLD_WIKIDATA_DATE="20210104"

# The New Wikidata JSON dump date to download (set it to a recent date of your choice)
# Precision : When downloading New wikidata, the script will look for the dump that is closest to this date. 
NEW_WIKIDATA_DATE="20230227"

# How many month to rewind in wikipedia to compute entity popularity
REWIND_N_MONTHS=12

# Database collection name
MONGODB_NAME='wiki'
