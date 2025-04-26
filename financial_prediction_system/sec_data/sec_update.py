import requests
import json
import pymongo
from datetime import datetime
import time
import os
import sys
from sqlalchemy import text
from financial_prediction_system.infrastructure.database.connection import SessionLocal
from financial_prediction_system.data_loaders.rate_limiter import rate_limited, with_retry
from financial_prediction_system.logging_config import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection (for company data)
# Option 1: Direct from .env file
# Use SQLAlchemy session for DB access
session = SessionLocal()

# Option 2: Using the config module (uncomment these lines if preferred)
# from financial_prediction_system.config import get_database_settings
# pg_conn = psycopg2.connect(get_database_settings().DATABASE_URL)
# pg_cursor = pg_conn.cursor()

# MongoDB connection
mongo_user = os.getenv("MONGO_USERNAME")
mongo_pass = os.getenv("MONGO_PASSWORD")
mongo_host = os.getenv("MONGO_HOST", "localhost") # Default to localhost if not set
mongo_port = os.getenv("MONGO_PORT", "27017") # Default to 27017 if not set
mongo_db_name = os.getenv("MONGO_DB_NAME", "sec_database") # Default db name

if mongo_user and mongo_pass:
    MONGO_URI = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/"
else:
    # Fallback to old connection string if no user/pass are set
    MONGO_URI = f"mongodb://{mongo_host}:{mongo_port}/"
    logger.warning("Connecting to MongoDB without authentication. Ensure MONGO_USERNAME and MONGO_PASSWORD are set in .env if authentication is enabled.")

client = pymongo.MongoClient(MONGO_URI)
sec_db = client[mongo_db_name]
companies = sec_db["companies"]
filings = sec_db["filings"]
facts = sec_db["facts"]
metrics = sec_db["metrics"]

# SEC API configuration
HEADERS = {
    "User-Agent": "Donovan don.pt33@gmail.com",
    "Accept-Encoding": "gzip, deflate"
}

@rate_limited(name="sec_api", tokens=1, tokens_per_second=5, max_tokens=5)
@with_retry(max_retries=3, base_delay=2.0, backoff_factor=2.0, 
           exceptions=(requests.RequestException, json.JSONDecodeError))
def get_with_retry(url):
    """Get data with SEC API rate limiting and exponential backoff retry"""
    logger.info(f"Requesting: {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()  # This will raise an exception for 4XX/5XX responses
    return response.json()

def get_company_filings(cik):
    """Get recent filings for a company by CIK number"""
    cik_formatted = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_formatted}.json"
    return get_with_retry(url)

def get_company_facts(cik):
    """Get financial statement data using the company facts API"""
    cik_formatted = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_formatted}.json"
    return get_with_retry(url)

def extract_key_metrics(facts_data, cik, symbol, filing_date, company_name, sector, industry):
    """Extract key financial metrics from facts data"""
    if not facts_data or 'facts' not in facts_data:
        return []

    metrics_list = []

    # Handle nested structure: facts['facts']['us-gaap']
    facts_facts = facts_data['facts'].get('facts')
    if facts_facts and isinstance(facts_facts, dict):
        # New structure: facts['facts']['us-gaap']
        us_gaap = facts_facts.get('us-gaap', {})
    else:
        # Fallback to old structure: facts['facts']['us-gaap']
        us_gaap = facts_data['facts'].get('us-gaap', {})

    # Key metrics to extract - expand as needed
    key_metrics = [
        'Revenue', 'NetIncome', 'GrossProfit', 'OperatingIncome',
        'Assets', 'Liabilities', 'StockholdersEquity',
        'EarningsPerShareBasic', 'EarningsPerShareDiluted',
        'DividendsPerShareCommonStockDeclared', 'OperatingExpenses',
        'CashAndCashEquivalentsAtCarryingValue', 'Goodwill',
        'AccountsReceivableNetCurrent', 'Inventory'
    ]

    for metric_name in key_metrics:
        if metric_name in us_gaap:
            metric_data = us_gaap[metric_name]
            # Extract the units (USD, shares, etc.)
            for unit_type, values in metric_data.get('units', {}).items():
                for value in values:
                    # Create a metric document
                    metric_doc = {
                        "cik": cik,
                        "symbol": symbol,
                        "company_name": company_name,
                        "sector": sector,
                        "industry": industry,
                        "metric": metric_name,
                        "value": value.get('val'),
                        "unit": unit_type,
                        "form": value.get('form'),
                        "filed_date": value.get('filed'),
                        "period_end": value.get('end'),
                        "period_start": value.get('start'),
                        "accn": value.get('accn'),
                        "fiscal_year": value.get('fy'),
                        "fiscal_period": value.get('fp'),
                        "captured_at": datetime.now()
                    }
                    metrics_list.append(metric_doc)
    return metrics_list


def process_company(symbol, company_name, sector, industry, missing_ciks=None):
    """Process a company's SEC data, only processing new filings and accumulating missing CIKs."""
    print(f"Processing {company_name} ({symbol})...")

    # Find CIK using the official company_tickers.json file
    cik = None
    try:
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(tickers_url, headers=HEADERS)
        response.raise_for_status()
        company_data = response.json()
        
        # Iterate through the dictionary values to find the matching ticker
        for item in company_data.values():
            if item.get('ticker', '').upper() == symbol.upper():
                cik = str(item.get('cik_str')) # CIK is already zero-padded
                break
                
    except requests.RequestException as e:
        print(f"Error fetching company tickers JSON: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing company tickers JSON: {e}")
    except Exception as e: # Catch any other unexpected errors during lookup
        print(f"An unexpected error occurred during CIK lookup for {symbol}: {e}")


    if not cik:
        print(f"Could not find CIK for {symbol}")
        if missing_ciks is not None:
            missing_ciks.append(symbol)
        return

    # Store company info
    company_doc = {
        "cik": cik,
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "updated_at": datetime.now()
    }
    companies.update_one(
        {"cik": cik},
        {"$set": company_doc},
        upsert=True
    )
    print(f"Updated company info for {symbol}")

    # Get latest filing accession_number in DB for this cik
    latest_filing = filings.find_one({"cik": cik}, sort=[("filing_date", -1)])
    latest_accn = latest_filing["accession_number"] if latest_filing else None
    latest_date = latest_filing["filing_date"] if latest_filing else None

    # Get company filings
    try:
        filings_data = get_company_filings(cik)
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Could not get filings for {symbol} (CIK: {cik}): {e}")
        filings_data = None # Ensure filings_data is None if request fails
    except Exception as e: # Catch other potential errors like JSONDecodeError
        logger.error(f"Unexpected error getting filings for {symbol} (CIK: {cik}): {e}")
        filings_data = None
        
    if not filings_data:
        print(f"No filings data retrieved for {symbol}")
        # Decide if you want to return here or continue to try getting facts
        # return # Optional: uncomment to stop processing if filings fail

    # Store only new filings
    if filings_data and 'filings' in filings_data and 'recent' in filings_data['filings']:
        recent_filings = filings_data['filings']['recent']
        stored_count = 0
        for i in range(len(recent_filings.get('accessionNumber', []))):
            accession_number = recent_filings['accessionNumber'][i] if i < len(recent_filings.get('accessionNumber', [])) else None
            filing_date = recent_filings['filingDate'][i] if i < len(recent_filings.get('filingDate', [])) else None
            # Only process new filings
            if latest_accn and accession_number <= latest_accn:
                continue
            filing_doc = {
                "cik": cik,
                "symbol": symbol,
                "company_name": company_name,
                "form_type": recent_filings['form'][i] if i < len(recent_filings.get('form', [])) else None,
                "filing_date": filing_date,
                "accession_number": accession_number,
                "file_number": recent_filings['fileNumber'][i] if i < len(recent_filings.get('fileNumber', [])) else None,
                "items": recent_filings['items'][i] if i < len(recent_filings.get('items', [])) else None,
                "size": recent_filings['size'][i] if i < len(recent_filings.get('size', [])) else None,
                "isXBRL": recent_filings['isXBRL'][i] if i < len(recent_filings.get('isXBRL', [])) else None,
                "isInlineXBRL": recent_filings['isInlineXBRL'][i] if i < len(recent_filings.get('isInlineXBRL', [])) else None,
                "primaryDocument": recent_filings['primaryDocument'][i] if i < len(recent_filings.get('primaryDocument', [])) else None,
                "primaryDocDescription": recent_filings['primaryDocDescription'][i] if i < len(recent_filings.get('primaryDocDescription', [])) else None,
                "fetched_at": datetime.now()
            }
            # Only count as new if not already present
            if not filings.find_one({"accession_number": accession_number}):
                filings.insert_one(filing_doc)
                stored_count += 1
            else:
                filings.update_one(
                    {"accession_number": accession_number},
                    {"$set": filing_doc}
                )
        print(f"Stored {stored_count} truly new filings for {symbol}")

    # Get facts data (only once per company), handle 404 gracefully
    facts_data = None
    try:
        facts_data = get_company_facts(cik)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"No Company Facts data found (404) for {symbol} (CIK: {cik}). Skipping facts processing.")
        else:
            # Re-raise other HTTP errors to be handled by the retry logic or general error handling
            logger.error(f"HTTP error getting facts for {symbol} (CIK: {cik}): {e}")
            raise # Or handle differently if needed
    except Exception as e:
        # Catch other potential errors (like JSONDecodeError from retry failures)
        logger.error(f"Error getting or processing facts for {symbol} (CIK: {cik}): {e}")
        # facts_data remains None

    if facts_data:
        # Store the whole facts data in the facts collection, sanitizing large ints
        def sanitize_for_mongo(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_mongo(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_mongo(item) for item in obj]
            elif isinstance(obj, int):
                if obj > 2**63 - 1 or obj < -2**63:
                    return str(obj)
                else:
                    return obj
            else:
                return obj

        sanitized_facts = sanitize_for_mongo(facts_data)
        facts.update_one(
            {"cik": cik},
            {"$set": {
                "cik": cik,
                "symbol": symbol,
                "company_name": company_name,
                "facts": sanitized_facts,
                "updated_at": datetime.now()
            }},
            upsert=True
        )
        print(f"Stored facts data for {symbol}")
        # Extract and store metrics
        filings_date = datetime.now().strftime("%Y-%m-%d")
        metrics_list = extract_key_metrics(
            facts_data, cik, symbol, filings_date,
            company_name, sector, industry
        )
        if metrics_list:
            # Insert metrics only if not already present (by cik, symbol, metric, period_end, value)
            # Use our bulk operations utility
            from financial_prediction_system.sec_data.bulk_operations import bulk_upsert_documents
            
            # Define key fields for uniqueness
            key_fields = ["cik", "symbol", "metric", "period_end", "value"]
            
            # Perform bulk upsert
            result = bulk_upsert_documents(metrics, metrics_list, key_fields)
            print(f"Stored {len(metrics_list)} metrics for {symbol} (upserted: {result['upserted']})")
    # Using the rate limiter decorator handles this for us now
    # No need for manual sleep


def test_symbol(symbol, missing_ciks=None):
    """Run the process for a single test symbol"""
    try:
        result = session.execute(
            text("SELECT symbol, company_name, sector, industry FROM stocks WHERE symbol = :symbol"),
            {"symbol": symbol}
        ).fetchone()
        if result:
            symbol, company_name, sector, industry = result
            process_company(symbol, company_name, sector, industry, missing_ciks)
        else:
            print(f"Symbol {symbol} not found in database. Using placeholder data for testing.")
            company_name = f"Test Company ({symbol})"
            sector = "Technology"
            industry = "Software"
            process_company(symbol, company_name, sector, industry, missing_ciks)
    except Exception as e:
        print(f"Error while testing symbol {symbol}: {e}")
    finally:
        pass


# Main process
def main(test_symbol_val=None, start_index=0):
    missing_ciks = []
    processed_symbols = set()
    start_time = datetime.now()
    
    # State file for resume functionality
    state_file = "sec_update_state.json"
    
    try:
        if test_symbol_val:
            print(f"Testing single symbol: {test_symbol_val}")
            test_symbol(test_symbol_val, missing_ciks)
        else:
            # Load already processed symbols to avoid reprocessing
            for company in companies.find({}, {"symbol": 1}):
                processed_symbols.add(company.get("symbol"))
            
            print(f"Already processed {len(processed_symbols)} symbols")
            
            # Use text() for the raw SQL query
            stocks = session.execute(text("SELECT symbol, company_name, sector, industry FROM stocks WHERE is_active = TRUE")).fetchall()
            for idx, (symbol, company_name, sector, industry) in enumerate(stocks):
                if idx < start_index:
                    continue
                
                if symbol in processed_symbols:
                    print(f"Skipping already processed {symbol}")
                    continue
                    
                try:
                    process_company(symbol, company_name, sector, industry, missing_ciks)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    # Log the error and continue with next symbol
                    with open("error_log.txt", "a") as f:
                        f.write(f"{datetime.now()} - Error processing {symbol}: {e}\n")
                
                # Save progress periodically
                if idx % 10 == 0:
                    with open(state_file, "w") as f:
                        json.dump({
                            "resume_from": symbol,
                            "index": idx,
                            "missing_ciks": missing_ciks,
                            "processed_count": idx + 1,
                            "last_updated": datetime.now().isoformat()
                        }, f)
                    print(f"Progress: {idx}/{len(stocks)} companies processed")
    finally:
        elapsed = datetime.now() - start_time
        print(f"\nProcess completed in {elapsed}")
        if missing_ciks:
            print("\nSymbols with missing CIKs:")
            print(", ".join(missing_ciks))
            with open("missing_ciks.txt", "w") as f:
                f.write("\n".join(missing_ciks))
        else:
            print("\nNo missing CIKs!")
        session.close()


if __name__ == "__main__":
    # Add resume functionality
    if len(sys.argv) > 1 and sys.argv[1].lower() == "resume":
        try:
            with open("sec_update_state.json", "r") as f:
                state = json.load(f)
                start_index = state.get("index", 0)
            print(f"Resuming from index {start_index}")
            main(start_index=start_index)
        except FileNotFoundError:
            print("No state file found. Starting from beginning.")
            main()
    elif len(sys.argv) > 1:
        test_symbol_val = sys.argv[1].upper()
        main(test_symbol_val)
    else:
        main()