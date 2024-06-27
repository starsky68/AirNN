import numpy as np
import matplotlib.pyplot as plt
x=[1,6,12,24]
# Beijing
# y1=[30.39,37.77, 44.98,52.18]
# y2=[27.45,34.09,40.62,47.02]
# y3=[25.83,33.13,40.6,47.9 ]
# y4=[24.66,31.69, 37.02,43.97]
# Jingjinji
y1=[12.32,19.95,21.93,23.42]
y2=[12.93,14.75,16.5,19.43]
y3=[10.45,12.84,14.79,16.44]
y4=[9.07,10.55,11.53,12.17]


my_x=[1,3,6,9,12,15,18,21,24]
plt.grid(True, linestyle="-.", linewidth=0.5)
plt.xticks(my_x)
plt.plot(x, y2, lw=1.5, c='darkorange', marker='s', ms=5, label='GCRNN')  # 绘制y1
plt.plot(x, y1, lw=1.5, c='darkgreen', marker='o', ms=5, label='DCRNN')  # 绘制y1
plt.plot(x, y3, lw=1.5, c='dodgerblue', marker='^', ms=5, label='GCGRU')  # 绘制y1
plt.plot(x, y4, lw=1.5, c='blue', marker='D', ms=5, label='YM4AQP')  # 绘制y1
# l1=plt.plot(x,y1,'-o',c='darkgreen',label='DCRNN')
# l2=plt.plot(x,y2,'-s',c='darkgreen',label='GCRNN')
# l3=plt.plot(x,y3,'-^c',c='darkgreen',label='GCGRU')
# l4=plt.plot(x,y4,'-*m',c='darkgreen',label='YM4AQP')
# plt.plot(x,y1,'-o',x,y2,'-sr',x,y3,'-^c',x,y4,'-*m')
plt.title('MAE on  Jingjinji')
plt.xlabel('Prediction length (h)')
plt.ylabel('MAE')
plt.legend()
plt.savefig("../figures/JJJ_MAE_Baseline.png", dpi=300)
plt.show()
