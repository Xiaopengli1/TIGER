#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str, default="../data/Beauty/item_emb.parquet", help="Input data path.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    parser.add_argument('--use_post_linear', action='store_true', help='Enable shared linear layer after quantization')
    parser.add_argument('--no_post_linear_bias', action='store_true', help='Disable bias term in the shared linear layer')
    
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt/Beauty", help="please specify output directory for model")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  use_post_linear=args.use_post_linear,
                  post_linear_bias=not args.no_post_linear_bias,
                  )
    print(model)

    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=64,
        shuffle=False,
        pin_memory=True
    )

    # 根据模型层数动态生成前缀与层名
    num_layers = len(args.num_emb_list)
    prefix = make_prefix(num_layers)  # e.g. ["<a_{}>","<b_{}>",...]
    layer_names = [p[1:3].split('_')[0] for p in prefix]  # 近似提取 a,b,c,... 用于打印

    # ======== 首轮编码（use_sk=False） ========
    all_indices = []
    all_indices_str = []

    for d in tqdm(data_loader, desc="Encoding (no sk)"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)          # [B, L] 的整型索引
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code_tokens = []
            for i, ind in enumerate(index):
                code_tokens.append(prefix[i].format(int(ind)))
            all_indices.append(code_tokens)
            all_indices_str.append(str(code_tokens))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)
    all_indices_shape = all_indices.shape
    print(f"all_indices_str_length: {all_indices_shape}")

    # ======== 只保留末层随机性，迭代去冲突 ========
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0

    tt = 0
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"[Round {tt}] collision groups: {len(collision_item_groups)}")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code_tokens = []
                for i, ind in enumerate(index):
                    code_tokens.append(prefix[i].format(int(ind)))
                all_indices[item] = code_tokens
                all_indices_str[item] = str(code_tokens)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate:", (tot_item - tot_indice) / tot_item)

    # ======== 将码串转为整数矩阵，并追加“去重位” ========
    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}

    codes = []
    for key, value in all_indices_dict.items():
        # "<a_12>" -> 12
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)
    codes_array = np.array(codes, dtype=int)

    # 追加一列去重位（初始 0）
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    # 若整行依然重复（含最后一列），则递增最后一列以强制唯一
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        print("Resolving duplicates in codes...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i  # 末位序号 0,1,2,...

    # 再检验一次
    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]
    if len(duplicates) > 0:
        print("There still have duplicates:", duplicates)
    else:
        print("There are no duplicates in the codes after resolution.")

    # ======== 统计每层指标（不包含去重位）并保存 ========
    per_layer_indices = codes_array[:, :-1]  # shape: (N, L)
    metrics = compute_layer_metrics(
        per_layer_indices,
        num_emb_list=args.num_emb_list,
        layer_names=layer_names,
        print_table=True
    )

    metrics_file = os.path.splitext(output_file)[0] + "_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {metrics_file}")

    # ======== 保存码本结果 ========
    print(f"Saving codes to {output_file}")
    print(f"the first 5 codes: {codes_array[:5]}")
    np.save(output_file, codes_array)


if __name__ == "__main__":
    main()
