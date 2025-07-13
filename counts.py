import re
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as gf
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

cog1 = [0.1, 0.1902, 0.1968, 0.2238, 0.2301, 0.2324, 0.2409, 0.3037, 0.3296, 0.3426, 0.3424, 0.3368, 0.326, 0.3485, 0.3778, 0.4383, 0.363, 0.3896, 0.4301, 0.4018, 0.3746, 0.3918, 0.3902, 0.4179, 0.4333, 0.4952, 0.4867, 0.4356, 0.4417, 0.4526, 0.4313, 0.4866, 0.4594, 0.4443, 0.3794, 0.4934, 0.4694, 0.49, 0.529, 0.4169, 0.4768, 0.4267, 0.5177, 0.4737, 0.5344, 0.5214, 0.5038, 0.5226, 0.4806, 0.5285, 0.5678, 0.5757, 0.5618, 0.5713, 0.5676, 0.5581, 0.5799, 0.576, 0.5447, 0.5537, 0.5685, 0.5638, 0.5779, 0.5347, 0.5757, 0.5889, 0.5353, 0.5445, 0.5606, 0.5635, 0.5758, 0.5842, 0.588, 0.5879, 0.5756, 0.5749, 0.574, 0.5755, 0.5262, 0.5545, 0.5476, 0.5686, 0.5702, 0.5789, 0.5774, 0.5787, 0.5677, 0.4787, 0.5607, 0.5792, 0.5559, 0.5801, 0.554, 0.5579, 0.5877, 0.5638, 0.592, 0.5617, 0.5467, 0.581, 0.5691, 0.591, 0.5923, 0.5637, 0.5735, 0.597, 0.5525, 0.5887, 0.5455, 0.531, 0.504, 0.5707, 0.5654, 0.5847, 0.5639, 0.5702, 0.5758, 0.5297, 0.578, 0.5631, 0.5876, 0.5788, 0.5863, 0.5889, 0.5868, 0.5629, 0.5915, 0.5631, 0.5309, 0.5941, 0.5936, 0.5942, 0.5976, 0.5877, 0.5829, 0.5746, 0.538, 0.6197, 0.5377, 0.5891, 0.5745, 0.5288, 0.6045, 0.5343, 0.6138, 0.5901, 0.581, 0.5826, 0.6209, 0.5296, 0.5778, 0.5887, 0.5965, 0.5605, 0.5775, 0.5758, 0.5706, 0.5891, 0.5724, 0.5523, 0.6044, 0.5963, 0.595, 0.565, 0.5842, 0.5608, 0.5917, 0.5916, 0.5571, 0.6092, 0.5358, 0.6019, 0.5902, 0.6065, 0.5267, 0.5876, 0.5671, 0.5905, 0.5539, 0.6045, 0.5934, 0.5794, 0.5624, 0.5994, 0.6108, 0.5851, 0.5722, 0.6198, 0.6005, 0.5623, 0.5921, 0.5378, 0.5731, 0.5993, 0.5894, 0.6102, 0.6045, 0.5774, 0.5924, 0.6091]
cog2 = [0.01, 0.01, 0.0174, 0.0303, 0.0376, 0.0531, 0.0643, 0.0798, 0.0826, 0.0919, 0.0926, 0.1114, 0.1183, 0.1241, 0.1305, 0.1253, 0.1079, 0.1404, 0.1536, 0.1506, 0.1605, 0.1617, 0.1702, 0.1697, 0.1682, 0.1722, 0.1733, 0.1609, 0.1705, 0.1793, 0.1868, 0.1697, 0.1651, 0.1859, 0.1898, 0.1825, 0.1866, 0.1713, 0.1864, 0.1921, 0.1912, 0.1944, 0.1822, 0.1891, 0.1591, 0.1915, 0.1878, 0.1669, 0.1831, 0.1894, 0.2078, 0.1961, 0.2077, 0.2043, 0.2029, 0.2075, 0.0131, 0.0076, 0.0234, 0.0325, 0.0484, 0.046, 0.0667, 0.0657, 0.074, 0.0805, 0.0922, 0.0836, 0.097, 0.1106, 0.1119, 0.0708, 0.1003, 0.1093, 0.1155, 0.1258, 0.1303, 0.1249, 0.1228, 0.1343, 0.0683, 0.1195, 0.139, 0.1405, 0.1475, 0.1455, 0.1163, 0.0989, 0.1471, 0.1557, 0.159, 0.1639, 0.162, 0.1676, 0.1662, 0.1717, 0.1732, 0.1722, 0.1486, 0.1741, 0.1738, 0.1789, 0.1849, 0.1695, 0.1831, 0.1784, 0.1847, 0.189, 0.183, 0.1935, 0.1962, 0.187, 0.2022, 0.1909, 0.195, 0.2001, 0.2002, 0.1981, 0.1953, 0.2061, 0.1966, 0.2015, 0.1938, 0.204, 0.2022, 0.1847, 0.177, 0.2086, 0.2073, 0.1993, 0.2066, 0.1944, 0.2052, 0.2017, 0.2111, 0.215, 0.2047, 0.2127, 0.206, 0.2093, 0.2098, 0.2112, 0.2168, 0.2171, 0.2062, 0.2125, 0.2183, 0.2143, 0.2155, 0.2102, 0.2122, 0.2183, 0.2203, 0.2158, 0.217, 0.2056, 0.2146, 0.2192, 0.2108, 0.2052, 0.2198, 0.2179, 0.2242, 0.2192, 0.2105, 0.2271, 0.2198, 0.2132, 0.2261, 0.2114, 0.2179, 0.2202, 0.221, 0.223, 0.2239, 0.2261, 0.2251, 0.2165, 0.2182, 0.231, 0.2248, 0.2329, 0.2194, 0.2116, 0.2241, 0.2268, 0.1946, 0.2138, 0.2327, 0.2274, 0.23, 0.2309, 0.229, 0.2252, 0.2331, 0.223, 0.2384, 0.2301, 0.2342, 0.2255]
fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# axins = inset_axes(ax, width="40%", height="30%",loc='center left',
#                    bbox_to_anchor=(0.5, 0.1, 1, 1),
#                    bbox_transform=ax.transAxes)
pitch_ = 1

class simsImagePlt(object):
    def __init__(self, file_path, color=None, style=None, label=None, marker=None) -> None:
        self.path = file_path
        self.color = color
        self.style = style
        self.label = label
        self.marker = marker

    def __count_key_values__(self):
        sims_list = []
        try:
            with open(self.path, 'r') as file:
                for line in file:
                    match = re.search(r'similarity list:\s*(\{.*?\})', line)
                    if match:
                        sims_list.append(match.group(1))
        except FileNotFoundError:
            print(f"Error: The file '{self.path}' was not found.")
        except Exception as e:
            print(f"Error: {e}")
        # print(sims_list)
        return sims_list

    def plt_image(self):
        marker = self.marker
        data_list = self.__count_key_values__()
        # matrix = np.array([list(ast.literal_eval(d).values()) for d in data_list])
        # avg_matrix = matrix.reshape(200, 10, 10).mean(axis=1)  # (200, 10)
        # print(avg_matrix.shape)
        avg_matrix = []
        for i in range(0, len(data_list), 10):
            div = [0] * 10
            group = data_list[i:i+10]  # 取出当前 10 个字典
    
        # 初始化一个临时字典，用于累加 0-9 的值
            temp_sum = {k: 0.0 for k in range(10)}
    
        # 遍历当前组的 10 个字典，累加值
            for d in group:
                tmp = ast.literal_eval(d)
                for key in range(10):
                    if tmp[key] != 0:
                        div[key] += 1
                    temp_sum[key] += tmp[key]
    
            # 计算均值，并存入 result
            avg_dict = {k: temp_sum[k] / div[k] if div[k] else 0.5 for k in range(10)}
            avg_matrix.append(list(avg_dict.values()))
        # print(len(avg_matrix[0]))
        avg_matrix = np.array(avg_matrix)
        plt.figure(figsize=(14, 6))

        # 绘制 10 条曲线（每条代表一个下标 0-9 的变化）
        for idx in range(10):
            # print(avg_matrix[:, idx])
            plt.plot(np.arange(0, 200, 10), gf(avg_matrix[0:200:10, idx], sigma=0.5), label=f"class {idx}", marker='o')

        # 添加标签、标题、图例
        plt.xlabel("Communication Round", fontsize=14, weight='bold')
        plt.ylabel("Similarity(β=0.1)", fontsize=14, weight='bold')
        # plt.xticks(np.arange(0,200,20))
        plt.title("Top accuracy: 57.23%(T)/53.27%(S) without adjustment", fontsize=14, weight='bold')
        plt.legend(loc="upper right", ncol=10)  # 图例放在右侧
        # plt.grid(True, linestyle="--", alpha=0.6)  # 网格线
        plt.tight_layout()  # 避免标签重叠
        plt.savefig('img_output/sims_01_noniid.pdf')
        plt.show()
        
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
        # print(accuracies)
        # print(t_accuracies)
        # print(np.max(accuracies))
        if self.flag:
            t_top_acc = np.max(t_accuracies[:])
            print(t_top_acc)
        
        if self.flag:
            return accuracies, t_accuracies
        else:
            return accuracies
    # t_accuracies if self.flag else accuracies
    
    def plt_image(self):
        marker, sigma, f = self.marker, 1.0, False
        round = np.arange(self.rs, self.re, self.pitch)
        if self.flag:
            accuracies, t_accuracies = self.__count_key_values__()
            acces = accuracies[self.rs:self.re:self.pitch]
            t_acces = t_accuracies[self.rs:self.re:self.pitch]
        else:
            accuracies = self.__count_key_values__()
            acces = accuracies[self.rs:self.re:self.pitch]
        if not self.flag:
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
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:s8.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=1.0_remark:molight_05_10.log',
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_gkd___.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=1.0_remark:molight_05_10.log',
    ]
    avgm_paths = [
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_m.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.1_remark:base_m.log',
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_m.log',
    ]
    prox_paths = [
        base_path + 'fedprox_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'fedprox_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'fedprox_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base.log',
    ] 
    moon_paths = [
        base_path + 'moon_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=100_frac=0.5_remark:molight_05_05.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=100_frac=1.0_remark:molight_05_10.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=50_frac=1.0_remark:base.log',
        base_path + 'moon_set=cifar10_beta=0.5_client=100_frac=1.0_remark:molight_05_10.log',
    ]
    disco_paths = [
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_disco.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=50_frac=0.1_remark:disco_client_50.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=150_frac=0.1_remark:disco_client_150.log',
    ]  
    gkd_paths = [
        base_path + 'fedavg_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base_gkd.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=100_frac=0.2_remark:gkd_frac_02.log',
        base_path + 'fedavg_set=cifar10_beta=0.5_client=150_frac=0.1_remark:gkd_client_150.log',
    ]  
    mine_paths = [
        base_path + 'mine_set=cinic10_beta=0.5_client=100_frac=0.1_remark:base.log',
        base_path + 'mine_set=cifar10_beta=0.5_client=50_frac=1.0_remark:sims_test.log',
        base_path + 'mine_set=cifar10_beta=0.5_client=100_frac=1.0_remark:sims_test.log',
        base_path + 'mine_set=cifar10_beta=0.1_client=100_frac=0.1_remark:sims_test_noniid.log',
        base_path + 'mine_set=cifar10_beta=0.5_client=100_frac=0.1_remark:sims_test_noniid.log',
        base_path + 'mine_set=cifar10_beta=5.0_client=100_frac=0.1_remark:sims_test_noniid.log'
    ]

    # fedavg_05_01 = ImagePlt(avg_paths[0], 0, 200, pitch_, color=['lightgreen'], style=['-'], marker='.', label=['FedAvg_C=0.1'])
    # fedavg_05_05 = ImagePlt(avg_paths[1], 0, 200, pitch_, color=['lightgreen'], style=['-'], marker='o', label=['FedAvg_C=0.5'])
    # fedavg_05_10 = ImagePlt(avg_paths[2], 0, 200, pitch_, color=['lightgreen'], style=['-'], marker='x', label=['FedAvg_C=1.0'])
    # moon_05_01 = ImagePlt(moon_paths[0], 0, 200, pitch_, color=['blue'], style=['-'], marker='.', label=['MOON_C=0.1'])
    # moon_05_05 = ImagePlt(moon_paths[1], 0, 200, pitch_, color=['blue'], style=['-'], marker='o', label=['MOON_C=0.5'])
    # moon_05_10 = ImagePlt(moon_paths[2], 0, 200, pitch_, color=['blue'], style=['-'], marker='x', label=['MOON_C=1.0'])
    # mine_05_01 = ImagePlt(mine_paths[0], 0, 196, pitch_, ismine=True, color=['lightgreen', 'green'], style=['-', '-'], label=['ours_C=0.1;w/o all', 'ours_C=0.1;w/ all'])
    # mine_05_05 = ImagePlt(mine_paths[1], 0, 200, pitch_, ismine=True, color=['black', 'blue'], style=['-', '-'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    # mine_05_10 = ImagePlt(mine_paths[0], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['-', '-'], label=['Ours_C=0.1(S)', 'Ours_C=0.1(T)'])
    # mine_05_10_ = ImagePlt(mine_paths[3], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['--', '--'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    # mine_pitch_10 = ImagePlt(mine_paths[4], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['-', '-'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    # mine_pitch_10_ = ImagePlt(mine_paths[5], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['--', '--'], label=['ours_C=0.2;w/o all', 'ours_C=0.2;w/ all'])
    fedavg = ImagePlt(avg_paths[0], 0, 200, pitch_, color=['lightgreen'], style=['-'], label=['FedAvg'])
    fedavg_ = ImagePlt(avg_paths[1], 0, 200, pitch_, color=['orange'], style=['-'], label=['FedCOG'])
    # fedavg__ = ImagePlt(avg_paths[1], 0, 30, pitch_, color=['blue'], style=['-'], label=['FedGKD+_7.0'])
    # fedavg___ = ImagePlt(avg_paths[1], 0, 30, pitch_, color=['black'], style=['-'], label=['FedGKD+_12.0'])
    fedavgm = ImagePlt(avgm_paths[0], 0, 200, pitch_, color=['dodgerblue'], style=['-'], label=['FedAvgM'])
    fedprox = ImagePlt(prox_paths[0], 0, 200, pitch_, color=['black'], style=['-'], label=['FedProx'])
    moon = ImagePlt(moon_paths[0], 0, 200, pitch_, color=['blue'], style=['-'], label=['MOON'])
    gkd = ImagePlt(gkd_paths[0], 0, 200, pitch_, color=['gray'], style=['-'], label=['FedGKD'])
    disco = ImagePlt(disco_paths[0], 0, 200, pitch_, color=['purple'], style=['-'], label=['FedDisco'])
    mine = ImagePlt(mine_paths[0], 0, 200, pitch_, ismine=True, color=['orange', 'red'], style=['-', '-'], label=['Ours(S)', 'Ours'])
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
    # mine_pitch_10.plt_image()
    # mine_pitch_10_.plt_image()
    fedavg.plt_image()
    
    # fedavg__.plt_image()
    # fedavg___.plt_image()
    fedavgm.plt_image()
    fedprox.plt_image()
    moon.plt_image()
    gkd.plt_image()
    disco.plt_image()
    fedavg_.plt_image()
    # ax.plot(np.arange(0, 200, 1), gf([acc * 100 for acc in cog1], sigma=1.0), c='orange', 
    #                 linestyle='-', marker=None, label='FedCOG')
    mine.plt_image()
    ax.legend(fontsize=12, loc="lower right", ncol=2)
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=0.5)
    # arr = np.append(np.arange(0, 50, 10), np.arange(50, 71, 2))
    ax.set_ylabel("Test Accuracy (%)", fontsize=15, weight='bold')
    ax.set_xlabel("Communication Round", fontsize=15, weight='bold')
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_yticks(np.arange(10, 51, step=10))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.savefig('img_output/cinic10.svg')
    # plt.show()
    
    # sims_mine = simsImagePlt(mine_paths[5])
    # sims_mine.plt_image()


