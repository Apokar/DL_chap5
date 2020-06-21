# -*- coding: utf-8 -*-
# @Time    : 2020/6/14 22:25
# @Author  : Apokar
# @Email   : Apokar@163.com
# @File    : tete.py
# @Comment :
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import matplotlib as mpl

ts = pd.Series(random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()


