import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# axins = inset_axes(ax, width="40%", height="30%",loc='center left',
#                    bbox_to_anchor=(0.5, 0.1, 1, 1),
#                    bbox_transform=ax.transAxes)
pitch_ = 10

class ImagePlt(object):
    def __init__(self, file_path, rs, re, pitch, ismine=False, color=None, style=None, label=None, marker=None) -> None:
        self.path = file_path
        self.flag = ismine
        self.rs = rs
        self.re = re
        self.pitch = pitch
        self.color = color
        self.style = style
        self.label = label
        self.marker = marker
        
    def __count_key_values__(self):
        accuracies = []
        t_accuracies = []
        try:
            with open(self.path, 'r') as file:
                for line in file:
                    match = re.search(r'Global Model Test accuracy: (\d+\.\d+)', line)
                    t_match = re.search(r'Teacher Model Test accuracy: (\d+\.\d+)', line)
                    if match:
                        acc = float(match.group(1))
                        accuracies.append(acc)
                    if t_match:
                        t_acc = float(t_match.group(1))
                        t_accuracies.append(t_acc)
        except FileNotFoundError:
            print(f"Error: The file '{self.path}' was not found.")
        except Exception as e:
            print(f"Error: {e}")
        print(accuracies)
        print(t_accuracies)
        print(np.max(accuracies))
        if self.flag:
            t_top_acc = np.max(t_accuracies[:])
            print(t_top_acc)
        
        if self.flag:
            return accuracies, t_accuracies
        else:
            return accuracies
    # t_accuracies if self.flag else accuracies
    
    def plt_image(self):
        marker, sigma, f = self.marker, 0.5, False
        round = np.arange(self.rs, self.re, self.pitch)
        if self.flag:
            accuracies, t_accuracies = self.__count_key_values__()
            acces = accuracies[self.rs:self.re:self.pitch]
            t_acces = t_accuracies[self.rs:self.re:self.pitch]
        else:
            accuracies = self.__count_key_values__()
            acces = accuracies[self.rs:self.re:self.pitch]
            
        ax.plot(round, gf([acc * 100 for acc in acces], sigma=sigma), c=self.color[0], 
                 linestyle=self.style[0], marker=marker, label=self.label[0])
        if f:
            axins.plot(round, gf([acc * 100 for acc in acces], sigma=sigma), c=self.color[0], 
                        linestyle=self.style[0], marker=marker, label=self.label[0])
        print("*" * 10)
            
        if self.flag:
            assert len(self.color) > 1 and len(self.style) > 1 and len(self.label) > 1, 'Error!'
            ax.plot(round, gf([t_acc * 100 for t_acc in t_acces], sigma=sigma), c=self.color[1], 
                 linestyle=self.style[1], marker=marker, label=self.label[1])
            if f:
                axins.plot(round, gf([t_acc * 100 for t_acc in t_acces], sigma=sigma), c=self.color[1], 
                    linestyle=self.style[1], marker=marker, label=self.label[1])
        if f:
            # 设置放大区间
            zone_left = 14
            zone_right = 17

            # 坐标轴的扩展比例（根据实际数据调整）
            x_ratio = 0.8 # x轴显示范围的扩展比例
            y_ratio = 0.6 # y轴显示范围的扩展比例

            # X轴的显示范围
            xlim0 = round[zone_left]-(round[zone_right]-round[zone_left])*x_ratio
            xlim1 = round[zone_right]+(round[zone_right]-round[zone_left])*x_ratio

            # Y轴的显示范围
            if self.flag:
                y = np.hstack((gf([acc * 100 for acc in acces], sigma=sigma)[zone_left:zone_right], 
                            gf([t_acc * 100 for t_acc in t_acces], sigma=sigma)[zone_left:zone_right]))
            else:
                y = gf([acc * 100 for acc in acces], sigma=sigma)[zone_left:zone_right]
            
            ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
            ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

            # 调整子坐标系的显示范围
            axins.set_xlim(xlim0, xlim1)
            axins.set_ylim(ylim0-1, ylim1)
        
        
if __name__ == '__main__':
    
    base_path = 'logs/experiment_log-'
    

    avg_paths = [
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.5_remark:frac_05.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=50_frac=0.1_remark:client_50.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=150_frac=0.1_remark:client_150.log',
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_gkd___.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=1.0_remark:molight_05_10.log',
    ]
    avgm_paths = [
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.1_remark:base_m.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.1_remark:base_m.log',
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_m.log',
    ]
    prox_paths = [
        base_path + 'fedprox_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'fedprox_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'fedprox_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base.log',
    ] 
    moon_paths = [
        base_path + 'moon_set=cifar10_beta=0.5_client=100_frac=0.5_remark:frac_05.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=50_frac=0.1_remark:client_50.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=150_frac=0.1_remark:client_150.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=50_frac=1.0_remark:base.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=100_frac=1.0_remark:molight_05_10.log',
    ]
    disco_paths = [
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.5_remark:disco_frac_05.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=50_frac=0.1_remark:disco_client_50.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=150_frac=0.1_remark:disco_client_150.log',
    ]  
    gkd_paths = [
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.5_remark:gkd_frac_05.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.2_remark:gkd_frac_02.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=150_frac=0.1_remark:gkd_client_150.log',
    ]  
    mine_paths = [
        base_path + 'mine_set=cifar10_beta=0.5_client=100_frac=0.5_remark:frac_05.log',
        base_path + 'mine_set=cifar10_beta=0.5_client=50_frac=0.1_remark:client_50.log',
        base_path + 'mine_set=cifar10_beta=0.5_client=150_frac=0.1_remark:clienr_150.log',
        base_path + 'mine_set=cinic10_beta=0.5_client=100_frac=0.1_remark:10_10.log',
        base_path + 'mine_set=cinic10_beta=5.0_client=100_frac=0.1_remark:base.log',
    ]

    # fedavg_05_01 = ImagePlt(avg_paths[0], 0, 200, pitch_, color=['lightgreen'], style=['-'], marker='.', label=['FedAvg_N=10'])
    # fedavg_05_05 = ImagePlt(avg_paths[1], 0, 200, pitch_, color=['lightgreen'], style=['-'], marker='o', label=['FedAvg_N=50'])
    # fedavg_05_10 = ImagePlt(avg_paths[2], 0, 200, pitch_, color=['lightgreen'], style=['-'], marker='x', label=['FedAvg_N=100'])
    # moon_05_01 = ImagePlt(moon_paths[0], 0, 200, pitch_, color=['red'], style=['-'], marker='.', label=['MOON_N=10'])
    # moon_05_05 = ImagePlt(moon_paths[1], 0, 200, pitch_, color=['red'], style=['-'], marker='o', label=['MOON_N=50'])
    # moon_05_10 = ImagePlt(moon_paths[2], 0, 200, pitch_, color=['red'], style=['-'], marker='x', label=['MOON_N=100'])
    # mine_05_01 = ImagePlt(mine_paths[0], 0, 196, pitch_, ismine=True, color=['lightgreen', 'green'], style=['-', '-'], label=['ours_C=0.1;w/o all', 'ours_C=0.1;w/ all'])
    # mine_05_05 = ImagePlt(mine_paths[1], 0, 200, pitch_, ismine=True, color=['black', 'blue'], style=['-', '-'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    # mine_05_10 = ImagePlt(mine_paths[2], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['-', '-'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    # mine_05_10_ = ImagePlt(mine_paths[3], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['--', '--'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    fedavg = ImagePlt(avg_paths[0], 0, 200, pitch_, color=['lightgreen'], style=['-'], label=['FedAvg'])
    # fedavg_ = ImagePlt(avg_paths[1], 0, 30, pitch_, color=['red'], style=['-'], label=['FedGKD+_10.0'])
    # fedavg__ = ImagePlt(avg_paths[1], 0, 30, pitch_, color=['blue'], style=['-'], label=['FedGKD+_7.0'])
    # fedavg___ = ImagePlt(avg_paths[1], 0, 30, pitch_, color=['black'], style=['-'], label=['FedGKD+_12.0'])
    # fedavgm = ImagePlt(avgm_paths[2], 0, 150, pitch_, color=['dodgerblue'], style=['-'], label=['FedAvgM'])
    # fedprox = ImagePlt(prox_paths[2], 0, 200, pitch_, color=['black'], style=['-'], label=['FedProx'])
    moon = ImagePlt(moon_paths[0], 0, 200, pitch_, color=['blue'], style=['-'], label=['MOON'])
    gkd = ImagePlt(gkd_paths[0], 0, 150, pitch_, color=['gray'], style=['-'], label=['FedGKD+'])
    disco = ImagePlt(disco_paths[0], 0, 150, pitch_, color=['purple'], style=['-'], label=['FedDisco'])
    mine = ImagePlt(mine_paths[0], 0, 150, pitch_, ismine=True, color=['orange', 'red'], style=['-', '-'], label=['Ours(S)', 'Ours(T)'])
    # fedavg_05_01.plt_image()
    # fedavg_05_05.plt_image()
    # fedavg_05_10.plt_image()
    # moon_05_01.plt_image()
    # moon_05_05.plt_image()
    # moon_05_10.plt_image()
    # mine_05_01.plt_image()
    # mine_05_05.plt_image()
    # mine_05_10.plt_image()
    # mine_05_10_.plt_image()
    # fedavg.plt_image()
    # fedavg_.plt_image()
    # fedavg__.plt_image()
    # fedavg___.plt_image()
    # fedavgm.plt_image()
    # fedprox.plt_image()
    # moon.plt_image()
    gkd.plt_image()
    disco.plt_image()
    mine.plt_image()

    ax.legend(fontsize=10, loc="lower right", ncol=2)
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=0.5)
    # arr = np.append(np.arange(0, 50, 10), np.arange(50, 71, 2))
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_xlabel("Communication Round", fontsize=14)
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_yticks(np.arange(0, 76, step=10))
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    # plt.savefig('img_output/motivation.jpg')
    # plt.show()
    


