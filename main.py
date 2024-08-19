from stock.stock_code_class import stockBase

stockName = stockBase("000001", ".sz")
print(stockName.codeNumStr, stockName.exchangeTailStr)
print(stockName.codeNameStr())

