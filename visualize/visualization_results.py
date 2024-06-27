import numpy as np
from matplotlib import pyplot as plt

def visualization_dcrnn_prediction(filename1: str,filename2: str,filename3: str,filename4: str):
    f1 = np.load(filename1)
    f2 = np.load(filename2)
    f3 = np.load(filename3)
    f4 = np.load(filename4)

    # print(f.files())
    # exit()
    prediction1 = f1["prediction"] # (12, 256, 74)
    prediction2 = f2["prediction"]  # (12, 256, 74)
    prediction3 = f3["prediction"]  # (12, 256, 74)
    prediction4 = f4["prediction"]  # (12, 256, 74)
    truth = f3["truth"] # (12, 256, 74)



    plt.Figure()
    plt.grid(True, linestyle="-.", linewidth=0.5)
    # plt.plot(prediction1[0, :240, 1],lw=1,c='#BD3C29')
    plt.plot(prediction4[1, :200,20],lw=1)
    # plt.plot(prediction2[0, :240, 1], lw=1,c='darkorange')
    # plt.plot(prediction2[0, :240, 35], lw=1)
    # plt.plot(prediction3[0, :120, 20],lw=1,  c='#e18727')
    # plt.plot(prediction4[0,:120, 20],lw=1, c='#0172B6')
    # # plt.plot(truth[0, :120, 35], c='darkgreen')
    # plt.plot(truth[0, :240,1],lw=1.2, c='#21854F')
    plt.plot(truth[0, :200,20],lw=1.2)
    # plt.plot(truth[0,:120,1],lw=1.2, c='darkgreen')
    # plt.legend(["YM4AQP_prediction","GCGRU_prediction", "DCRNN_prediction", "FCRNN_prediction","Truths"], loc="upper left")
    plt.title('1021A')
    plt.xlabel('Prediction time (h)')
    plt.ylabel('AQI')
    plt.legend(["ATGCN","Truths"], loc="upper left")
    # plt.legend(["YM4AQP_prediction","GCGRU_prediction","FCRNN_prediction","Truths"], loc="upper left")


    plt.savefig("../figures/1021_ATGCN.png", dpi=600)
    plt.show()



if __name__ == "__main__":
    visualization_dcrnn_prediction("../data/YM4AQP_BJ_drop0.3_CELL2_withoutsprase_transdrop0.4_droppathin_original_prediction.npz",
                                   "../data/YM4AQP_BJ_GCGRU_USERU_prediction.npz",
                                   "../data/DCRNN_BJ_RUFlase_prediction.npz",
                                   "../data/FCRNN_BJ_RUFlase_prediction.npz")
    # visualization_dcrnn_prediction("../data/YM4AQP_GANSU_bnn_random_walk_drop0_CELL2_withoutsprase_4transdrop0_droppathin_sq_prediction.npz",
    #                                "../data/GCGRU_GansuQX_prediction.npz",
    #                                "../data/DCRNN_BJ_RUFlase_prediction.npz",
    #                                "../data/FCRNN_BJ_RUFlase_prediction.npz")
