# coding: utf-8
# @email: XXXX@mail.com
r"""
SiMGR
################################################
Reference:
    [SiMGR: Sign-aware Multi-modal Graph Recommendation]
"""
import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
# import faiss
from sklearn.cluster import KMeans
from common.abstract_recommender import GeneralRecommender
from utils.myutils import RegLoss, MYEmbLoss
from torch_geometric.utils import spmm
import torch.nn.functional as F
import os
from utils.utils import build_sim,build_graph_from_adj


class AttentionAugmentation(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionAugmentation, self).__init__()
        # 定义可学习参数
        self.query = nn.Parameter(torch.randn(embedding_dim))  # 查询向量
        self.key_positive = nn.Linear(embedding_dim, 1)  # 正嵌入的键变换
        self.key_negative = nn.Linear(embedding_dim, 1)  # 负嵌入的键变换

    def forward(self, e, e_pos, e_neg):
        """
        参数:
        - e: 原始嵌入 (batch_size, embedding_dim)
        - e_pos: 正嵌入 (batch_size, embedding_dim)
        - e_neg: 负嵌入 (batch_size, embedding_dim)

        返回:
        - 增广后的嵌入 (batch_size, embedding_dim)
        """
        # batch_size, embedding_dim = e.size()

        # 计算注意力分数
        score_pos = self.key_positive(e_pos)  # (batch_size, 1)
        score_neg = self.key_negative(e_neg)  # (batch_size, 1)

        # 归一化注意力分数
        attention_scores = torch.cat([score_pos, score_neg], dim=1)  # (batch_size, 2)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, 2)

        # 加权求和
        weighted_pos = attention_weights[:, 0].unsqueeze(1) * e_pos  # (batch_size, embedding_dim)
        weighted_neg = attention_weights[:, 1].unsqueeze(1) * e_neg  # (batch_size, embedding_dim)
        augmented_embedding = e + F.relu(weighted_pos + weighted_neg)  # 增广嵌入
        return augmented_embedding




class SiMGR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SiMGR, self).__init__(config, dataset)
        self.a_feat = None   ###no audio feature
        self.config = config
        self.sparse = True
        self.reg_weight = config['reg_weight']
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_ui_layers = config['n_ui_layers']
        self.n_mm_layer = config['n_mm_layer']
        self.n_nodes = self.n_users + self.n_items

        self.info_pos = config['info_pos']
        self.info_neg = config['info_neg']
        self.cl_loss = config['cl_loss']
        self.cl_loss2 = config['cl_loss2']



        self.knn_k = config['knn_k']
        self.knn_i = config['knn_i']
        self.knn_a = config['knn_a']
        self.tail_ratio = config['tail_ratio']
        self.positive_threshold = config['positive_threshold']
        self.negative_threshold = config['negative_threshold']

        #####same with SiGRec
        self.cos_margin = config['cos_margin'] if 'cos_margin' in config else 0.9
        self.lambda2 = config['lambda2'] if 'lambda2' in config else 1.0
        self.neg_weight = config['neg_weight'] if 'neg_weight' in config else 200
        self.restore_user_e, self.restore_item_e = None, None


        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.dataset_path = dataset_path
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)  ##同时保留未拆分的交互图

        self.pos_inter_mat = dataset.pos_inter_mat
        self.neg_inter_mat = dataset.neg_inter_mat

        ####Define user/item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)



        self.u2thresh_dict = dataset.user_thresh_map

        self.pos_interaction_matrix, self.neg_interaction_matrix = dataset.signed_inter_matrix(form='coo')
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.pos_adj = self.scipy_matrix_to_sparse_tenser(self.pos_interaction_matrix,  torch.Size((self.n_users, self.n_items)))
        self.neg_adj = self.scipy_matrix_to_sparse_tenser(self.neg_interaction_matrix,  torch.Size((self.n_users, self.n_items)))

        self.num_inters_pos = self.get_norm_adj_mat_new2(self.pos_interaction_matrix)
        self.num_inters_neg = self.get_norm_adj_mat_new2(self.neg_interaction_matrix)
        self.num_inters_pos = torch.FloatTensor(1.0 / (self.num_inters_pos + 1e-7)).to(self.device)
        self.num_inters_neg = torch.FloatTensor(1.0 / (self.num_inters_neg - 1e-7)).to(self.device)

        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        self.pos_norm_adj_matrix = dataset.pos_norm_adj_matrix.to(self.device)
        self.neg_norm_adj_matrix = dataset.neg_norm_adj_matrix.to(self.device)


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)

            self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)

            self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))

        self.is_popular_mask = np.load(os.path.join(dataset_path, f'pop_item_msk-{1 - self.tail_ratio}.npy'))
        self.is_tail_mask = np.load(os.path.join(dataset_path, f'tail_item_msk-{self.tail_ratio}.npy'))
        self.is_popular_mask = torch.FloatTensor(self.is_popular_mask.reshape(-1, 1).squeeze()).to(self.device)
        self.is_tail_mask = torch.FloatTensor(self.is_tail_mask.reshape(-1, 1).squeeze()).to(self.device)

        image_adj = build_sim(self.image_embedding.weight.detach())
        text_adj = build_sim(self.text_embedding.weight.detach())
        mm_attractive = torch.zeros_like(self.norm_adj)
        mm_attractive_R = torch.zeros_like(self.R)
        un_mm_attractive = torch.zeros_like(self.norm_adj)

        for user, items in dataset.history_items_per_u.items():
            items = torch.tensor([i for i in items]).to(self.device)  #
            k_num = self.knn_a + items.size(0)
            #
            mm_sim = torch.multiply(image_adj[items], text_adj[items])  # (items_size, all_items)
            mm_all_pop_candidates = mm_sim * self.is_popular_mask
            mm_all_tail_candidates = mm_sim * self.is_tail_mask
            idx = 0
            for i in items:
                mm_all_tail_candidates[idx, i] = -999  # filter already interacted items
                mm_all_pop_candidates[idx, i] = 999  # filter already_interacted items, a smaller value
                idx += 1

            most_k_un_attract_value, most_k_un_attract_indices = torch.topk(-mm_all_pop_candidates, k_num, dim=-1)
            most_k_attract_value, most_k_attract_indices = torch.topk(mm_all_tail_candidates, k_num,
                                                                      dim=-1)  # 冷门物品要最相似的k个

            k_mm_value, k_mm_indices = torch.topk(most_k_attract_value.flatten(), k_num)

            selected_indices = torch.where(k_mm_value >= self.positive_threshold)
            k_mm_value = k_mm_value[selected_indices]
            k_mm_indices = k_mm_indices[selected_indices]
            edge_value = torch.ones(size=k_mm_value.size()).to(self.device)
            mm_indices = most_k_attract_indices.flatten()[k_mm_indices]

            uid = torch.zeros_like(mm_indices).fill_(user)  # (k_num)
            mm_sparse = torch.sparse_coo_tensor(torch.stack([uid, self.n_users + mm_indices]), edge_value,
                                                size=self.norm_adj.size())
            mm_sparse_t = torch.sparse_coo_tensor(torch.stack([self.n_users + mm_indices, uid]), edge_value,
                                                  size=self.norm_adj.size())
            mm_R_sparse = torch.sparse_coo_tensor(torch.stack([uid, mm_indices]), edge_value, size=self.R.size())
            mm_attractive += mm_sparse
            mm_attractive += mm_sparse_t
            mm_attractive_R += mm_R_sparse

            ### negative edge for popular items
            k_un_mm_value, k_un_mm_indices = torch.topk(most_k_un_attract_value.flatten(), k_num, dim=-1)

            un_selected_indices = torch.where(k_un_mm_value <= self.negative_threshold)
            k_un_mm_value = k_un_mm_value[un_selected_indices]
            k_un_mm_indices = k_un_mm_indices[un_selected_indices]
            most_k_un_attract_indices = most_k_un_attract_indices.flatten()[k_un_mm_indices]
            edge_value = torch.ones(size=k_un_mm_value.size()).to(self.device)
            uid = torch.zeros_like(k_un_mm_indices).fill_(user)  # (items_size, k_num)
            un_mm_sparse = torch.sparse_coo_tensor(torch.stack([uid, self.n_users + most_k_un_attract_indices]),
                                                   edge_value, size=self.norm_adj.size())
            un_mm_sparse_t = torch.sparse_coo_tensor(torch.stack([self.n_users + most_k_un_attract_indices, uid]),
                                                     edge_value, size=self.norm_adj.size())

            un_mm_attractive += un_mm_sparse
            un_mm_attractive += un_mm_sparse_t

        mm_attractive_adj = build_graph_from_adj(mm_attractive, is_sparse=self.sparse, norm_type='sym', mask=False)
        mm_attractive_adj_R = build_graph_from_adj(mm_attractive_R, is_sparse=self.sparse, norm_type='sym', mask=False)
        un_mm_attractive_adj = build_graph_from_adj(un_mm_attractive, is_sparse=self.sparse, norm_type='sym',
                                                    mask=False)

        del image_adj, text_adj, mm_attractive, mm_sparse, mm_sparse_t, (
            mm_R_sparse), mm_attractive_R, un_mm_attractive
        torch.cuda.empty_cache()

        self.mm_attractive_adj = mm_attractive_adj.cuda()
        self.mm_attractive_adj_R = mm_attractive_adj_R.cuda()
        self.pos_norm_adj_matrix = torch.add(self.pos_norm_adj_matrix, self.mm_attractive_adj)

        self.un_mm_attractive_adj = un_mm_attractive_adj.cuda()
        self.neg_norm_adj_matrix = torch.add(self.neg_norm_adj_matrix, self.un_mm_attractive_adj)



        self.emb_loss = MYEmbLoss()
        self.reg_loss = RegLoss()

        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)



        self.neg_w = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(),
            nn.PReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh()
        )
        self.atten_aug = AttentionAugmentation(self.embedding_dim)


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        #return torch.sparse.FloatTensor(indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        # return torch.sparse.FloatTensor(i, data, shape).to(self.device)
        return torch.sparse_coo_tensor(i, data, shape, dtype=torch.float32).to(self.device)


    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        R = self.interaction_matrix.tolil()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        temp = L.tolil()
        self.R = temp[:self.n_users, self.n_users:]
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    def get_norm_adj_mat_new2(self, input_mat):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = input_mat
        inter_M_t = input_mat.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        return sumArr

    # modality graph embedding
    def mge_overall(self, modal='v'):
        if modal == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        else:  ####str='t'
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats

    def get_norm_adj_mat_from_edge(self, edge_list):
        n_nodes = self.num_users + self.num_items
        user_np = edge_list[:, 0]
        item_np = edge_list[:, 1]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.coo_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        A  =  tmp_adj + tmp_adj.transpose()
        sumArr = A.sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).coalesce().to(self.device)
        return  SparseL


    def pyg_spmm(self, sp_mat, emb):
        if sp_mat.layout == torch.sparse_coo:
            sp_mat = sp_mat.to_sparse_csr()
        out = spmm(src=sp_mat, other=emb, reduce='sum')
        return out

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_id_embedding.weight
        #item_embeddings = self.item_multimodal_fusing()
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


    def forward_positive(self, nonsplitflag=True):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for l_idx in range(self.n_ui_layers):
            all_embeddings = self.pyg_spmm(self.pos_norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        if nonsplitflag:
            return lightgcn_all_embeddings
        else:
            user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
            return user_all_embeddings, item_all_embeddings

    def forward_negative(self):
        # 先调用user-item positive graph得到 z_u^+, z_i^-
        # user_all_embeddings, item_all_embeddings = self.forward_positive()
        all_embeddings = self.forward_positive() #####torch.cat([user_all_embeddings, item_all_embeddings], dim=0)

        ###把从正向交互得到的用户/物品嵌入，在negative graph上进行图卷积 z_u^-, z_i^-  GCN on user-item negative graph
        lightgcn_all_embeddings = self.pyg_spmm(self.neg_norm_adj_matrix, all_embeddings)
        # 只返回item_表示, MLP layser
        lightgcn_all_embeddings = self.neg_w(lightgcn_all_embeddings)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def forward_overall_graph(self, nonsplitflag=True):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for l_idx in range(self.n_ui_layers):
            all_embeddings = self.pyg_spmm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        if nonsplitflag:
            return lightgcn_all_embeddings
        else:
            user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,[self.n_users, self.n_items])
            return user_all_embeddings, item_all_embeddings


    def forward(self, train=True):
        user_embs, item_embs = self.forward_overall_graph(nonsplitflag=False)  ##完整的用户-物品二部图
        new_content = torch.cat([user_embs, item_embs], dim=0)


        user_pos, item_pos = self.forward_positive(nonsplitflag=False)
        content_pos_embs = torch.cat([user_pos, item_pos], dim=0)

        user_neg, item_neg = self.forward_negative()
        content_neg_embs = torch.cat([user_neg, item_neg], dim=0)
        content_embs = self.atten_aug(new_content, content_pos_embs, content_neg_embs)


        ##multi-modal information fusion
        v_feats, t_feats = self.mge_overall(modal='v'), self.mge_overall(modal='t')
        mge_embs = F.normalize(v_feats) + F.normalize(t_feats)   ###[n_nodes, emb]

        # Attention Fuser
        att_common = torch.cat([self.attention(v_feats), self.attention(t_feats)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * v_feats + weight_common[:, 1].unsqueeze(dim=1) * t_feats
        side_embs = (mge_embs  - common_embeds) / 3

        u_embs = content_embs[:self.n_users] + side_embs[:self.n_users]
        i_embs = content_embs[self.n_users:] + side_embs[self.n_users:]

        if train:
            return u_embs, i_embs, content_embs, side_embs
        else:
            return u_embs, i_embs


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)
        return bpr_loss

    def distance(self, x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        dis = xx_cc - 2 * xc
        return dis


    def calculate_loss(self, interaction, epoch, freq=20):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[0].to(torch.long)
        pos_item = interaction[1].to(torch.long)
        neg_item = interaction[2].to(torch.long)
        rating = interaction[3]
        user_embed, item_embed, content_embeds, side_embeds = self.forward()   ###[user_num, 128] , [item_num, 128]

        u_embeddings = user_embed[user]
        pos_embeddings = item_embed[pos_item]
        neg_embeddings = item_embed[neg_item]

        mf_loss = self.bpr_loss(u_embeddings, pos_embeddings, neg_embeddings)



        v_list = []
        for u in user.tolist():
            v = self.u2thresh_dict[u]
            v_list.append(v)
        pt_tensor = torch.tensor(v_list).to(self.device)
        is_neg = rating <= pt_tensor



        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_users, content_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        # item-item contrastive loss
        cl_loss = (self.InfoNCE(side_embeds_items[pos_item], content_items[pos_item], self.info_pos)
                   + self.InfoNCE(side_embeds_users[user], content_users[user], self.info_neg))
        # user-item contrastive loss
        cl_loss2 = (self.InfoNCE(u_embeddings, content_items[pos_item], self.info_pos)
                    + self.InfoNCE(u_embeddings, side_embeds_items[pos_item],self.info_pos))


        reg_loss = (self.emb_loss.calc_loss(u_embeddings, pos_embeddings, neg_embeddings)
                        + self.reg_loss.calc_loss(self.neg_w.parameters()))
        x1 = u_embeddings
        x2 = pos_embeddings
        x3 = neg_embeddings
        target1 = torch.ones(u_embeddings.shape[0]).to(self.device)
        assert len(rating) == len(target1)
        target1[is_neg] = target1[is_neg] * -1
        target2 = -1 * torch.ones(u_embeddings.shape[0]).to(self.device)
        loss2 = F.cosine_embedding_loss(x1, x2, target1, margin=self.cos_margin, reduction='none')
        weight = torch.ones(u_embeddings.shape[0]).to(self.device)
        weight[is_neg] = self.neg_weight
        loss2 = torch.einsum("i,i->i", [loss2, weight]).mean()
        loss3 = self.neg_weight * F.cosine_embedding_loss(x1, x3, target2, margin=0.9)

        total_loss = (mf_loss + self.lambda2 * (loss2 + loss3) + self.reg_weight * reg_loss
                      + self.cl_loss * cl_loss + self.cl_loss2 * cl_loss2)

        return total_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.einsum('ij, kj->ik', [u_embeddings, i_embeddings])
        return scores

    def full_sort_predict(self, interaction):  ###for validation and testing
        user = interaction[0]
        self.restore_user_e, self.restore_item_e = self.forward(train=False)
        u_embeddings = self.restore_user_e[user]
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores
