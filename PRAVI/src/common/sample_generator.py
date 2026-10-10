# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import torch


def samples_gen(user_rep, item_rep, k):
    # Assume user_rep and item_rep are already defined with correct dimensions
    # user_rep: (A, 128), item_rep: (B, 128)

    # Step 1: Normalize vectors
    user_rep = user_rep / user_rep.norm(dim=1, keepdim=True)
    item_rep = item_rep / item_rep.norm(dim=1, keepdim=True)

    # Step 2: Compute cosine similarity matrix
    pos_similarity_matrix = torch.mm(user_rep, item_rep.t())  # Output shape: (A, B)
    neg_similarity_matrix = -pos_similarity_matrix
    
    # Step 3: Find the positions of the top-k most similar items for each user
    _, top_k_pos_indices = torch.topk(pos_similarity_matrix, k=k, dim=1)
    _, top_k_neg_indices = torch.topk(neg_similarity_matrix, k=k, dim=1)

    # top_k_pos_indices: positions of top-k positive samples for each user, shape (A, k)
    # top_k_neg_indices: positions of top-k negative samples for each user, shape (A, k)
    return top_k_pos_indices, top_k_neg_indices