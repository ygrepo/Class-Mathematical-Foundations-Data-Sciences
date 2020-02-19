import csv

import pynance as pn

names = [
    'aapl',  # Apple Inc, Sector: Information Technology
    'amzn',  # Amazon.con Inc, Sector: Information Technology
    'msft',  # Microsoft Corp, Sector: Information Technology
    'goog',  # Alphabet Inc, Sector: Information Technology
    'xom',  # Exxon Mobil Corp, Sector: Energy
    'apc',  # Anadarko Petroleum Corp, Sector: Energy
    'cvx',  # Chevron, Sector: Energy
    'c',  # Citigroup, Sector: Financial
    'gs',  # Goldman Sachs Group, Sector: Financial
    'jpm',  # JPMorgan Chase & Co, Sector: Financial
    'aet',  # Aetna Inc, Sector: Health Care
    'jnj',  # Johnson & Johnson, Sector: Health Care
    'dgx',  # Quest Diagnostics, Sector: Health Care
    'spy',  # State Street's SPDR S&P 500 ETF.  A security that roughly tracks
    # the S&P 500, a weighted average of the stock prices of
    # 500 top US companies.
    'xlf',  # State Street's SPDR Financials ETF.  A security that tracks
    # a weighted average of top US financial companies.
    'sso',  # ProShares levered ETF that roughly corresponds to twice
    # the daily performance of the S&P 500.
    'sds',  # ProShares inverse levered ETF that roughly corresponds to
    # twice the negative daily performance of the S&P 500.  That is,
    # when the S&P 500 goes up by a dollar, this roughly goes down by 2.
    'uso',  # Exchange traded product that tracks the price of oil in the US.
]

prices = dict()
for name in names:
    df = pn.data.get(name, '2016-01-01', '2017-09-20')
    prices[name] = df['Adj Close'].values
    prices['Dates'] = df.index.values
columns = ['Dates'] + names

with open('stockprices.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(zip(*[prices[key] for key in columns]))
