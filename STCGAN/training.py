# Training

from tqdm import tqdm
from utils import *
from network import *


def train_sc(feat_sc, nfeat, params=None):
    feat_sc = torch.FloatTensor(feat_sc).cuda()
    model_sc = SCC(nfeat).cuda()
    optimizer_sc = torch.optim.Adam([
        {'params': model_sc.parameters(), 'lr': 0.001, },
    ])
    for e in tqdm(range(params.epochs_sc)):
        model_sc.train()
        emb, out = model_sc(feat_sc)
        re_loss = F.mse_loss(out, feat_sc, reduction='mean')
        loss = re_loss
        optimizer_sc.zero_grad()
        loss.backward()
        optimizer_sc.step()
        if e == params.epochs_sc - 1:
            np.save(params.savepath + "sc_X.npy", out.cpu().detach().numpy())


def train_st(st_data=None, sc_data=None, graph_dict=None, params=None):
    ncell = sc_data.shape[0]
    nspot = st_data.shape[0]
    input_dim = st_data.shape[1]

    st_data = torch.FloatTensor(st_data).cuda()
    sc_data = torch.FloatTensor(sc_data).cuda()

    adj_label = graph_dict["adj_label"].cuda()
    edge_index = graph_dict["edge_index"].cuda()

    label_CSL = torch.FloatTensor(add_contrastive_label(nspot)).cuda()
    criterion = nn.BCEWithLogitsLoss()
    model_st = ST(input_dim, 512, 64).cuda()
    advnet = Discriminator(in_feature=64, hidden_size=32).cuda()
    model_map = Encoder_map(ncell, nspot).cuda()

    optimizer_st = torch.optim.Adam(model_st.parameters(), lr=0.001)
    optimizer = torch.optim.Adam([
        {'params': model_st.parameters(), 'lr': 0.0005},
        {'params': advnet.parameters(), 'lr': 0.001},
    ])
    optimizer_map = torch.optim.Adam([
        {'params': model_map.parameters(), 'lr': 0.001},
    ])
    #########################################################################################
    for e in tqdm(range(params.epochs_st)):
        model_st.train()
        z, re_st, readj, mu, logvar = model_st(st_data, edge_index)
        re_loss = F.mse_loss(re_st, st_data, reduction='mean')
        loss_gcn = gcn_loss(preds=readj, labels=adj_label, mu=mu,
                            logvar=logvar, n_nodes=nspot, norm=graph_dict["norm_value"],
                            mask=adj_label)
        loss = 10 * re_loss + 0.1 * loss_gcn
        optimizer_st.zero_grad()
        loss.backward()
        optimizer_st.step()
    #########################################################################################
    for e in tqdm(range(params.epochs_adv)):
        z, re_st, readj, mu, logvar = model_st(st_data, edge_index)
        b_z, _, _, bmu, blogvar = model_st(re_st, edge_index)

        re_loss = F.mse_loss(re_st, st_data, reduction='mean') + F.mse_loss(b_z, z, reduction='mean')
        loss_gcn = gcn_loss(preds=readj, labels=adj_label, mu=mu,
                            logvar=logvar, n_nodes=nspot, norm=graph_dict["norm_value"],
                            mask=adj_label)
        KLD = -0.5 / nspot * torch.mean(torch.sum(
            1 + 2 * blogvar - bmu.pow(2) - blogvar.exp().pow(2), 1))

        prob_source = advnet.forward(z.data, b_z.data)
        adv_loss = criterion(prob_source, label_CSL)
        total_loss = params.w_re1 * re_loss + params.w_gcn * loss_gcn + params.w_kld * KLD + params.w_adv1 * adv_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    #########################################################################################
    z, re_st, _, _, _ = model_st(st_data, edge_index)
    emb_sp = F.normalize(re_st.data, p=2, eps=1e-12, dim=1)
    emb_sc = F.normalize(sc_data, p=2, eps=1e-12, dim=1)
    model_st.eval()

    for e in tqdm(range(params.epochs_map)):
        model_map.train()
        map_matrix = model_map()
        map_probs = F.softmax(map_matrix, dim=1)
        pred_sp = torch.matmul(map_probs.t(), emb_sc)

        pz, _, _, _, _ = model_st(pred_sp, edge_index)

        prob_source = advnet.forward(z.data, pz)
        adv_loss = criterion(prob_source, label_CSL)

        loss_recon = F.mse_loss(pred_sp, emb_sp, reduction='mean')
        loss_NCE = regularization_loss(pred_sp, adj_label, 1 - adj_label)

        total_loss = params.w_re2 * loss_recon + params.w_nce * loss_NCE  + params.w_adv2 * adv_loss

        optimizer_map.zero_grad()
        total_loss.backward()
        optimizer_map.step()
        if e == params.epochs_map - 1:
            np.save(params.savepath + "Map.npy", map_probs.cpu().detach().numpy())
            np.save(params.savepath + "ST_X.npy", emb_sp.cpu().detach().numpy())
            np.save(params.savepath + "ST_z.npy", z.cpu().detach().numpy())
            np.save(params.savepath + "P_ST.npy", z.cpu().detach().numpy())