import datetime

class stockData:

    # 股票基本数据构造函数
    def __init__(self):
        self.stockName = None
        self.stockInfoDate = datetime.datetime.today()
        self.marketValue = 0
        self.closePrice = 0
        self.openPrice = 0
        self.highestPrice = 0
        self.lowestPrice = 0
        self.changePercent = 0
        self.amplitudePercent = 0
