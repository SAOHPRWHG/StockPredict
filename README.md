# StockPredict
This is a Data Mining Course Final Project.

本项目用于研究学习预测单支股票的日收盘价和涨跌。

以上证50ETF为例：

![2021-1-11](https://github.com/SAOHPRWHG/StockPredict/blob/main/fig/2021-1-11.png)

## 安装与使用

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