import requests
import json
import pymongo
from datetime import datetime
import time
from financial_prediction_system.infrastructure.database.connection import SessionLocal
import os
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
client = pymongo.MongoClient("mongodb://localhost:27017/")
sec_db = client["sec_database"]
companies = sec_db["companies"]
filings = sec_db["filings"]
facts = sec_db["facts"]
metrics = sec_db["metrics"]

# SEC API configuration
HEADERS = {
    "User-Agent": "Donovan don.pt33@gmail.com",
    "Accept-Encoding": "gzip, deflate"
}

def get_with_retry(url, max_retries=3):
    """Get data with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Requesting: {url}")
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too Many Requests
                wait_time = 10 * (attempt + 1)
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error {response.status_code}: {response.text}")
                time.sleep(5)
                return None
        except Exception as e:
            print(f"Exception during request: {e}")
            time.sleep(5)
    return None

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

    # First, find CIK for the company
    ticker_lookup_url = "https://www.sec.gov/include/ticker.txt"
    response = requests.get(ticker_lookup_url, headers=HEADERS)
    cik = None
    if response.status_code == 200:
        for line in response.text.splitlines():
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0].lower() == symbol.lower():
                cik = parts[1]
                break

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
    filings_data = get_company_filings(cik)
    if not filings_data:
        print(f"No filings data for {symbol}")
        return

    # Store only new filings
    if 'filings' in filings_data and 'recent' in filings_data['filings']:
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

    # Get facts data (only once per company)
    facts_data = get_company_facts(cik)
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
            inserted_count = 0
            for metric_doc in metrics_list:
                query = {
                    "cik": metric_doc["cik"],
                    "symbol": metric_doc["symbol"],
                    "metric": metric_doc["metric"],
                    "period_end": metric_doc["period_end"],
                    "value": metric_doc["value"]
                }
                if not metrics.find_one(query):
                    metrics.insert_one(metric_doc)
                    inserted_count += 1
            print(f"Stored {inserted_count} new metrics for {symbol}")
    # Respect SEC rate limits
    time.sleep(1)


def test_symbol(symbol, missing_ciks=None):
    """Run the process for a single test symbol"""
    try:
        result = session.execute(
            "SELECT symbol, company_name, sector, industry FROM stocks WHERE symbol = :symbol",
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
def main(test_symbol_val=None):
    missing_ciks = []
    try:
        if test_symbol_val:
            print(f"Testing single symbol: {test_symbol_val}")
            test_symbol(test_symbol_val, missing_ciks)
        else:
            stocks = session.execute("SELECT symbol, company_name, sector, industry FROM stocks WHERE is_active = TRUE").fetchall()
            for symbol, company_name, sector, industry in stocks:
                process_company(symbol, company_name, sector, industry, missing_ciks)
                time.sleep(2)
    finally:
        if missing_ciks:
            print("\nSymbols with missing CIKs:")
            print(", ".join(missing_ciks))
        else:
            print("\nNo missing CIKs!")
        session.close()


if __name__ == "__main__":
    import sys
    
    # Check if a symbol was provided as a command-line argument
    if len(sys.argv) > 1:
        test_symbol_val = sys.argv[1].upper()
        main(test_symbol_val)
    else:
        main()