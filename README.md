# StockPredict
This is a Data Mining Course Final Project.

本项目用于研究学习预测单支股票的日收盘价和涨跌。

以上证50ETF为例：

![2021-1-11](https://github.com/SAOHPRWHG/StockPredict/blob/main/fig/2021-1-11.png)

![2021-1-12](https://github.com/SAOHPRWHG/StockPredict/blob/main/fig/2021-1-12.png)

## 安装

预测模型依赖如下库文件：

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
import talib
from jqdatasdk import *
from jqdatasdk import finance
from sklearn import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import dateutil
```

其中tensorflow，talib和jqdata一般需要独立安装

其中tensorflow版本为1.15

安装请参考官网教程

https://www.tensorflow.org/install?hl=zh-cn

jqdata

```
pip install jqdatasdk
```

TA_lib

由于TA_lib二进制版文件仅有32位，建议从以下网址下载至本地安装

https://www.lfd.uci.edu/~gohlke/pythonlibs/

以我的python3.7为例，下载对应版本TA_Lib‑0.4.19‑cp37‑cp37m‑win_amd64.whl文件至本地目录

在控制台中进入该目录

```
pip install TA_Lib‑0.4.19‑cp37‑cp37m‑win_amd64.whl
```

即可安装成功。

## 使用

在本地目录中打开控制台

```
python3 StockPredict.py
```

待运行完毕即可获得预测结果：

![2021-1-11](https://github.com/SAOHPRWHG/StockPredict/blob/main/fig/2021-1-11.png)

predict.png为较长时间段的拟合曲线。