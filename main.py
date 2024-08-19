from stock.stock_base_class import stockBase
from stock.stock_data_class import stockData
import pandas as pd
from WindPy import w


stockBase = stockBase("000001", ".sz")
#print(stockName.codeNumStr, stockName.exchangeTailStr)
#print(stockName.codeNameStr())

stock = stockData()
stock.stockName = stockBase.codeNameStr()

w.start()
w.isconnected()

data = pd.DataFrame()
data = w.wss(stock.stockName, "OPEN")

stock.openPrice = data.Data

print(stock.openPrice)
