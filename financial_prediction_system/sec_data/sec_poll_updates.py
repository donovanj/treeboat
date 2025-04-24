import time
import os
from datetime import datetime
import requests
import pymongo
import json
from financial_prediction_system.infrastructure.database.connection import SessionLocal
from financial_prediction_system.data_loaders.rate_limiter import rate_limited, with_retry
from financial_prediction_system.logging_config import logger
from dotenv import load_dotenv

def send_slack_notification(webhook_url, message):
    payload = {"text": message}
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        if resp.status_code != 200:
            print(f"Slack notification failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Slack notification error: {e}")

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")  # Set this in your .env
POLL_INTERVAL = int(os.getenv("SEC_POLL_INTERVAL", 600))  # seconds, default 10 min

# Use SQLAlchemy session for DB access
session = SessionLocal()

client = pymongo.MongoClient("mongodb://localhost:27017/")
sec_db = client["sec_database"]
filings = sec_db["filings"]

HEADERS = {
    "User-Agent": "Donovan don.pt33@gmail.com",
    "Accept-Encoding": "gzip, deflate"
}

@rate_limited(name="sec_api", tokens=1, tokens_per_second=0.1, max_tokens=1)
@with_retry(max_retries=3, base_delay=2.0, backoff_factor=2.0, 
           exceptions=(requests.RequestException,))
def get_cik_for_symbol(symbol):
    url = "https://www.sec.gov/include/ticker.txt"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        parts = line.strip().split('\t')
        if len(parts) == 2 and parts[0].lower() == symbol.lower():
            return parts[1]
    return None

def get_latest_filing_accession(cik):
    doc = filings.find_one({"cik": cik}, sort=[("filing_date", -1)])
    return doc["accession_number"] if doc else None

@with_retry(max_retries=3, base_delay=2.0, backoff_factor=2.0,
           exceptions=(requests.RequestException, json.JSONDecodeError))
def get_sec_data(url):
    """Get SEC data with retry logic"""
    logger.info(f"Requesting: {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

from financial_prediction_system.sec_data.bulk_operations import bulk_upsert_documents

def bulk_insert_filings(filings_docs):
    """Insert filings in bulk for better performance"""
    if not filings_docs:
        return 0
    
    # Use our bulk operations utility
    key_fields = ["accession_number", "cik"]
    result = bulk_upsert_documents(filings, filings_docs, key_fields)
    return result['upserted']

@rate_limited(name="sec_api", tokens=1, tokens_per_second=0.1, max_tokens=1)
def poll_and_update():
    symbols = [row[0] for row in session.execute("SELECT symbol FROM stocks WHERE is_active = TRUE").fetchall()]
    all_new_filings = []
    
    for symbol in symbols:
        try:
            cik = get_cik_for_symbol(symbol)
            if not cik:
                logger.warning(f"Could not find CIK for {symbol}")
                continue
                
            latest_accn = get_latest_filing_accession(cik)
            
            # Rate-limited SEC API request
            url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
            data = get_sec_data(url)
            
            filings_data = data.get('filings', {}).get('recent', {})
            new_filings_docs = []
            
            for i, accn in enumerate(filings_data.get('accessionNumber', [])):
                if latest_accn and accn <= latest_accn:
                    continue
                    
                filing_doc = {
                    "cik": cik,
                    "symbol": symbol,
                    "form_type": filings_data['form'][i] if i < len(filings_data.get('form', [])) else None,
                    "filing_date": filings_data['filingDate'][i] if i < len(filings_data.get('filingDate', [])) else None,
                    "accession_number": accn,
                    "fetched_at": datetime.now()
                }
                new_filings_docs.append(filing_doc)
                all_new_filings.append(filing_doc)
            
            # Bulk insert instead of one at a time
            if new_filings_docs:
                inserted = bulk_insert_filings(new_filings_docs)
                if inserted > 0:
                    logger.info(f"{datetime.now()}: {symbol} - {inserted} new SEC filings added.")
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Send notifications for all new filings at once
    if all_new_filings and SLACK_WEBHOOK_URL:
        for filing in all_new_filings:
            filing_msg = f"{filing['symbol']} ({filing['cik']}): {filing['form_type']} filed on {filing['filing_date']} (Acc#: {filing['accession_number']})"
            send_slack_notification(SLACK_WEBHOOK_URL, filing_msg)

if __name__ == "__main__":
    import sys
    
    # One-time run mode
    if len(sys.argv) > 1 and sys.argv[1].lower() == "once":
        logger.info(f"{datetime.now()}: Running SEC poll once")
        poll_and_update()
        logger.info(f"{datetime.now()}: Polling complete.")
        session.close()
        sys.exit(0)
    
    # Normal polling mode
    while True:
        try:
            logger.info(f"{datetime.now()}: Polling SEC for new filings...")
            poll_and_update()
            logger.info(f"{datetime.now()}: Polling complete. Sleeping {POLL_INTERVAL} seconds.")
            session.close()  # Close and reopen session each cycle
            session = SessionLocal()
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Error in polling loop: {e}")
            # Sleep and continue in case of error
            time.sleep(60)
            try:
                session.close()
                session = SessionLocal()
            except:
                pass
