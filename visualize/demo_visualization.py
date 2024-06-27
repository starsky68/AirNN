import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision

torchvision.models.resnet.resnet50()

# 显示中文
font = {'family':'serif', 'weight':'normal', 'size':'12'}
plt.rc('font', **font)               # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）


def get_temperature_data(file_name):
    temperature_data = np.load(file_name)

    return temperature_data


if __name__ == "__main__":
    temperature_data = get_temperature_data("../data/JINjinji_3provice/Jingjinji_poll/Jingjinji_data.npy")
    print(temperature_data.shape) # (86, 1296, 11)
    print(temperature_data)

    plt.plot(temperature_data[2, :180,:1], label='1001A', color="#5b9bd5")
    plt.plot(temperature_data[3, :180,:1], label='1029A', color="#ed7d31")
    plt.plot(temperature_data[4, :180,:1], label='1036A', color="#70ad47")

    plt.xlabel("Time/h")
    plt.ylabel("Pollution/C")
    plt.xticks(range(0, 181, 12))  # 设置x轴
    plt.legend()
    # plt.show()

    plt.savefig("../figures/jjj_pollution1.png", dpi=600)
    plt.savefig("../figures/jjj_pollution1.svg")
    plt.show()