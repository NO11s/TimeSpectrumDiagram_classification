import matplotlib.pyplot as plt
import numpy as np

def draw_curve(figname_list, data_list):
    n_figs = len(figname_list)
    for idx in range(n_figs):
        plt.subplot(1, n_figs, idx+1)
        plt.plot(np.array(data_list[idx]))
        plt.title(figname_list[idx])
    
    plt.savefig('./train_process.png')
    plt.show()