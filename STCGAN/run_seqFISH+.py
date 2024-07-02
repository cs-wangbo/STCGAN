from __future__ import division
from __future__ import print_function

import argparse
from training import *

parser = argparse.ArgumentParser()

parser.add_argument('--feat_hidden1', type=int, default=256, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=64, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--seed', type=int, default=100, help='seed.')
parser.add_argument('--epochs_sc', type=int, default=1000, help='epochs.')
parser.add_argument('--epochs_st', type=int, default=1000, help='epochs.')
parser.add_argument('--epochs_adv', type=int, default=500, help='epochs.')
parser.add_argument('--epochs_map', type=int, default=300, help='epochs.')
parser.add_argument('--w_re1', type=int, default=10, help='Reconstruct weight 1.')
parser.add_argument('--w_re2', type=int, default=1, help='Reconstruct weight 2.')
parser.add_argument('--w_gcn', type=int, default=1, help='gcn weight.')
parser.add_argument('--w_kld', type=int, default=0.01, help='kld weight.')
parser.add_argument('--w_adv1', type=int, default=1, help='Discriminator z weight 1.')
parser.add_argument('--w_adv2', type=int, default=0.01, help='Discriminator z weight 2.')
parser.add_argument('--w_nce', type=int, default=0.1, help='Regularization weight.')


parser.add_argument('--rp', type=int, default=0.05, help='retain_percent.')  # 0.05

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)
params = parser.parse_args()
params.device = device

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    datasets = ['10000']
    # '10000', '6000', '3000'
    for i in range(len(datasets)):

        dataset = datasets[i]
        print(dataset)
        path = "../generate_data/seqFISH/" + dataset + "/"

        if not os.path.exists('./result/seqFISH/'):
            os.mkdir('./result/seqFISH/')
        savepath = './result/seqFISH/' + dataset + '/'

        params.savepath = savepath

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        st_adata, sc_adata, graph_dict = load_data(path)
        fix_seed(params.seed)

        train_sc(feat_sc=sc_adata.X, nfeat=sc_adata.X.shape[1], params=params)
        sc_x = np.load(params.savepath + "sc_X.npy")
        train_st(st_data=st_adata.X, sc_data=sc_x, graph_dict=graph_dict, params=params)

        map_probs = np.load(params.savepath + "Map.npy")
        st_adata.obsm['map_matrix'] = map_probs.T
        project_cell_to_spot(st_adata, sc_adata, retain_percent=params.rp)
        df_projection = st_adata.obs[st_adata.obsm['proportion'].columns]

        rmse_results = rmse(st_adata.obsm['proportion'], df_projection)
        rmse_results = np.around(rmse_results, 3)
        jsd_results = distance.jensenshannon(st_adata.obsm['proportion'], df_projection)
        jsd_results = np.around(jsd_results, 3)

        results = pd.DataFrame([rmse_results, jsd_results], columns=st_adata.obsm['proportion'].columns,
                               index=['rmse', 'jsd'])
        print(results)
        print('mean rmse:', np.around(np.mean(rmse_results), 3))
        print('mean jsd:', np.around(np.mean(jsd_results), 3))


        import matplotlib

        matplotlib.use('Agg')
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap


        xl = st_adata.obsm['spatial'][:, 0] / 250
        yl = st_adata.obsm['spatial'][:, 1] / 500

        # 自定义颜色映射
        cmap_colors = [(1, 1, 1), (1, 0, 0)]  # 白色到红色
        cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        # 绘制网格图
        fig, ax = plt.subplots(figsize=(10, 4))  # 图像大小根据方格数量调整
        for i, j, val in zip(xl, yl, st_adata.obs['iNeuron']):
            color = val  # 0表示白色，1表示红色
            ax.fill([i, i + 2, i + 2, i, i],
                    [j, j, j + 1, j + 1, j], color=cmap(val),
                    edgecolor='black')  # 使用fill绘制矩形


        tmprmse = str(np.around(results.loc['rmse', 'iNeuron'], 2))
        tmpjsd = str(np.around(results.loc['jsd', 'iNeuron'], 2))

        title = dataset + ' genes (RMSE = ' + tmprmse + ', JSD = ' + tmpjsd + ')'

        ax.set_title(title)
        ax.axis('off')
        cax = fig.add_axes([0.1, 0.7, 0.3, 0.08])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Proportion')
        plt.show()
        plt.savefig(
            savepath + "STCGAN" + dataset + '.jpg',
            bbox_inches='tight',
            dpi=300)