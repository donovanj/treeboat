import pymongo
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
sec_db = client["sec_database"]

# Create collections
companies = sec_db["companies"]
filings = sec_db["filings"]
facts = sec_db["facts"]
metrics = sec_db["metrics"]
sectors = sec_db["sectors"]

# Create indexes for improved performance
companies.create_index([("cik", 1)], unique=True)
companies.create_index([("symbol", 1)], unique=True)
companies.create_index([("sector", 1)])
companies.create_index([("industry", 1)])

filings.create_index([("cik", 1)])  # For company lookups
filings.create_index([("form_type", 1)])  # For filtering by form type
filings.create_index([("filing_date", -1)])  # For sorting by date, descending
filings.create_index([("cik", 1), ("form_type", 1), ("filing_date", -1)])  # Compound index

facts.create_index([("cik", 1)], unique=True)  # One facts document per company

metrics.create_index([("cik", 1)])
metrics.create_index([("symbol", 1)])
metrics.create_index([("period_end", -1)])  # For time series analysis
metrics.create_index([("sector", 1), ("industry", 1)])  # For sector/industry analysis