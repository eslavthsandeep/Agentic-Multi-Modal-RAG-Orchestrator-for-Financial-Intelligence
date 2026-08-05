import logging
from datetime import datetime, timedelta

import yfinance as yf

from app.db.sql_client import execute_query, init_tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_stock_data() -> None:
    """Seed AAPL stock data into the SQLite database."""
    ticker = "AAPL"
    
    logger.info("Initializing database tables...")
    init_tables()
    
    # Check if data exists
    check_sql = "SELECT COUNT(*) as count FROM stock_history WHERE ticker = ?"
    result = execute_query(check_sql, (ticker,))
    if result["rows"][0]["count"] > 0:
        logger.info(f"Data for {ticker} already exists. Skipping seed.")
        return
        
    logger.info(f"Downloading 5 years of data for {ticker}...")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        
        if df.empty:
            logger.warning(f"No data downloaded for {ticker}")
            return
            
        insert_sql = """
            INSERT INTO stock_history 
            (ticker, date, open_price, high_price, low_price, close_price, volume) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        rows_inserted = 0
        for index, row in df.iterrows():
            params = (
                ticker,
                index.strftime("%Y-%m-%d"),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"])
            )
            execute_query(insert_sql, params)
            rows_inserted += 1
            
        logger.info(f"Successfully inserted {rows_inserted} rows of {ticker} data.")
    except Exception as e:
        logger.error(f"Failed to seed stock data: {e}")
        raise

if __name__ == "__main__":
    seed_stock_data()
