class stockBase:

    # 股票基类属性定义  init 600000.sh
    codeNumStr = "600000"
    exchangeTailStr = ".sh"

    # 构造函数
    def __init__(self, codeNumStr, exchangeTailStr):
        self.codeNumStr = codeNumStr
        self.exchangeTailStr = exchangeTailStr
    
    # 拼接交易所代码
    def codeNameStr(self):
        return "{}{}".format(self.codeNumStr,self.exchangeTailStr)


