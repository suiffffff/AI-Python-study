from turtledemo.chaos import plot

import matplotlib.pyplot as plt
from matplotlib import font_manager

my_font=font_manager.FontProperties(fname="/Windows/Fonts/SIMHEI.TTF")
fig=plt.figure(figsize=(20,8),dpi=80)

x=range(2,26,2)
y=[15,13,14.5,17,20,25,26,26,24,22,18,15]
y2=[14,16,18,20,22,24,26,28,30,25,20,15]

_xtick_labels=[f"{i}点" for i in range(2,26,2)]

plt.xticks(x,_xtick_labels,rotation=45,fontproperties=my_font)
plt.yticks(range(min(y),max(y)+1))

plt.xlabel("时间",fontproperties=my_font)
plt.ylabel("温度 单位°",fontproperties=my_font)

plt.plot(x,y,label="山城")
plt.plot(x,y2,label="蜀都")
plt.grid(alpha=0.4)
plt.legend(prop=my_font)
plt.savefig("./sig_size.png")

plt.show()