import yfinance as yf
from datetime import datetime
endDate = datetime.now().strftime('%Y %m %d').split()
startDate = endDate[:]
startDate[0] = str(int(startDate[0]) - 1)
stocks = ["XOM", "RRC", "BBY", "MA", "JPM", "PFE"]

for _ in stocks:
    stock_data = yf.download(_, start='-'.join(startDate), end='-'.join(endDate), group_by='column')
    stock_data.to_csv(f'./{_}.csv')
