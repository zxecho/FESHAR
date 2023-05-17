import matplotlib.pyplot as plt
import numpy as np


def plot_resutls(resutls, results_save_path, name):
    avg_acc, avg_auc, loss = resutls
    x_len = np.arange(len(avg_acc))

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    axs[0].plot(x_len, avg_acc)
    axs[0].set_xlabel('Conmunication#')
    axs[0].set_ylabel('Avg test acc')
    axs[0].grid(True)

    axs[1].plot(np.arange(len(avg_auc)), avg_auc)
    axs[1].set_xlabel('Conmunication#')
    axs[1].set_ylabel('Avg test auc')
    axs[1].grid(True)

    axs[2].plot(x_len, loss)
    axs[2].set_xlabel('Conmunication#')
    axs[2].set_ylabel('Loss')
    axs[2].grid(True)

    fig.tight_layout()
    plt.savefig(results_save_path + name + '.jpg')
    # plt.show()


if __name__ == '__main__':
    results = ([1, 2, 3], [1, 5, 6], [2, 3, 4])
    dir = './'
    name = 'test'
    plot_resutls(results, dir, name)
