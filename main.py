import datetime
from stock.stock_base_class import stockBase
from stock.stock_data_class import stockData
import pandas as pd
from WindPy import w

TodayDate = datetime.datetime.today().strftime("%Y-%m-%d")

stockBase = stockBase("000001", ".sz")
#print(stockName.codeNumStr, stockName.exchangeTailStr)
#print(stockName.codeNameStr())

stock = stockData()
stock.stockName = stockBase.codeNameStr()
stock.stockInfoDate = TodayDate
 
w.start()

stock.closePrice = w.wsd(stock.stockName, "close", "ED", stock.stockInfoDate, "PriceAdj=F", usedf=True)[1].iloc[0,0]
stock.openPrice = w.wsd(stock.stockName, "open", "ED", stock.stockInfoDate, "PriceAdj=F", usedf=True)[1].iloc[0,0]
stock.highestPrice = w.wsd(stock.stockName, "high", "ED", stock.stockInfoDate, "PriceAdj=F", usedf=True)[1].iloc[0,0]
stock.lowestPrice = w.wsd(stock.stockName, "low", "ED", stock.stockInfoDate, "PriceAdj=F", usedf=True)[1].iloc[0,0]

print(w.wsd(stock.stockName, "tech_MA5", "2024-08-10", "2024-08-20", "PriceAdj=F", usedf=True)[1])
w.close()

print(stock.lowestPrice, stock.highestPrice, stock.openPrice, stock.closePrice)