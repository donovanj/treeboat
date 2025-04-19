import time
import os
from datetime import datetime
import requests
from financial_prediction_system.infrastructure.database.connection import SessionLocal
import pymongo
from dotenv import load_dotenv

# For Slack integration
import json

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

def get_cik_for_symbol(symbol):
    url = "https://www.sec.gov/include/ticker.txt"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        for line in resp.text.splitlines():
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0].lower() == symbol.lower():
                return parts[1]
    return None

def get_latest_filing_accession(cik):
    doc = filings.find_one({"cik": cik}, sort=[("filing_date", -1)])
    return doc["accession_number"] if doc else None

def poll_and_update():
    symbols = [row[0] for row in session.execute("SELECT symbol FROM stocks WHERE is_active = TRUE").fetchall()]
    for symbol in symbols:
        cik = get_cik_for_symbol(symbol)
        if not cik:
            print(f"Could not find CIK for {symbol}")
            continue
        latest_accn = get_latest_filing_accession(cik)
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            print(f"Failed to fetch filings for {symbol}")
            continue
        data = resp.json()
        filings_data = data.get('filings', {}).get('recent', {})
        new_count = 0
        new_filings = []
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
            filings.insert_one(filing_doc)
            new_filings.append(filing_doc)
            new_count += 1
        if new_count:
            msg = f"{datetime.now()}: {symbol} - {new_count} new SEC filings added."
            print(msg)
            if SLACK_WEBHOOK_URL:
                for filing in new_filings:
                    filing_msg = f"{filing['symbol']} ({filing['cik']}): {filing['form_type']} filed on {filing['filing_date']} (Acc#: {filing['accession_number']})"
                    send_slack_notification(SLACK_WEBHOOK_URL, filing_msg)

if __name__ == "__main__":
    while True:
        print(f"{datetime.now()}: Polling SEC for new filings...")
        poll_and_update()
        print(f"{datetime.now()}: Polling complete. Sleeping {POLL_INTERVAL} seconds.")
        session.close()
        time.sleep(POLL_INTERVAL)
