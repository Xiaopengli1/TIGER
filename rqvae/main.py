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


# ========== 工具函数：冲突检测 ==========
def check_collision(all_indices_str: np.ndarray) -> bool:
    """是否所有码串都唯一"""
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str: np.ndarray):
    """各码串出现次数"""
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str: np.ndarray):
    """返回撞码样本的分组（每个分组是具有同一码串的一组样本下标）"""
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


# ========== 指标统计：冲突率 / 利用率 / 熵 ==========
def compute_layer_metrics(
    indices: np.ndarray,
    num_emb_list=None,
    layer_names=None,
    print_table: bool = True,
):
    """
    indices: np.ndarray, 形状 (N, L)，每行是一个样本的逐层索引（不包含最后的“去重位”）
    num_emb_list: list[int] 或 None，每层码本大小 K_l；若 None，则用 max(index)+1 近似
    layer_names: list[str] 或 None，仅用于打印名称
    print_table: 是否打印结果表格

    返回: dict，包含每层指标和汇总
    """
    indices = np.asarray(indices)
    N, L = indices.shape
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(L)]

    results = []
    totals = {
        "N": int(N),
        "L": int(L),
        "avg_collision_rate": 0.0,
        "avg_collision_item_rate": 0.0,
        "avg_utilization": 0.0 if num_emb_list is not None else None,
        "avg_entropy_bits": 0.0,
        "avg_norm_entropy": 0.0 if num_emb_list is not None else None,
        "avg_perplexity": 0.0,
    }

    for l in range(L):
        x = indices[:, l]
        uniq, counts = np.unique(x, return_counts=True)
        U = int(uniq.size)

        # 码本大小
        K = int(num_emb_list[l]) if num_emb_list is not None else int(x.max()) + 1

        # 指标
        collision_rate = 1.0 - (U / N)                         # 样本 vs 唯一索引
        collision_item_rate = counts[counts > 1].sum() / N     # 参与冲突的样本占比
        utilization = (U / K) if num_emb_list is not None else None

        p = counts / N
        H_bits = float(-(p * np.log2(p)).sum()) if U > 0 else 0.0
        H_norm = (H_bits / math.log2(K)) if num_emb_list is not None and K > 1 else None
        perplexity = float(2.0 ** H_bits)

        results.append({
            "layer": str(layer_names[l]),
            "N": int(N),
            "K": int(K),
            "unique_codes": int(U),
            "collision_rate": float(collision_rate),
            "collision_item_rate": float(collision_item_rate),
            "utilization": float(utilization) if utilization is not None else None,
            "entropy_bits": float(H_bits),
            "norm_entropy": float(H_norm) if H_norm is not None else None,
            "perplexity": float(perplexity),
        })

    # 汇总（算术平均）
    totals["avg_collision_rate"] = float(np.mean([r["collision_rate"] for r in results]))
    totals["avg_collision_item_rate"] = float(np.mean([r["collision_item_rate"] for r in results]))
    totals["avg_entropy_bits"] = float(np.mean([r["entropy_bits"] for r in results]))
    totals["avg_perplexity"] = float(np.mean([r["perplexity"] for r in results]))
    if num_emb_list is not None:
        totals["avg_utilization"] = float(np.mean([r["utilization"] for r in results]))
        norm_list = [r["norm_entropy"] for r in results if r["norm_entropy"] is not None]
        totals["avg_norm_entropy"] = float(np.mean(norm_list)) if norm_list else None

    if print_table:
        head = f"{'layer':<10}{'N':>8}{'K':>8}{'unique':>10}{'coll_rate':>12}{'item_coll':>12}{'util':>8}{'H(bits)':>12}{'H_norm':>10}{'pplx':>10}"
        print("\n=== Per-layer metrics ===")
        print(head)
        print("-" * len(head))
        for r in results:
            print(f"{r['layer']:<10}{r['N']:>8}{r['K']:>8}{r['unique_codes']:>10}"
                  f"{r['collision_rate']:>12.4f}{r['collision_item_rate']:>12.4f}"
                  f"{(r['utilization'] if r['utilization'] is not None else float('nan')):>8.4f}"
                  f"{r['entropy_bits']:>12.4f}"
                  f"{(r['norm_entropy'] if r['norm_entropy'] is not None else float('nan')):>10.4f}"
                  f"{r['perplexity']:>10.2f}")
        print("\nAverages:")
        print(f"- avg_collision_rate      : {totals['avg_collision_rate']:.4f}")
        print(f"- avg_collision_item_rate : {totals['avg_collision_item_rate']:.4f}")
        if totals["avg_utilization"] is not None:
            print(f"- avg_utilization         : {totals['avg_utilization']:.4f}")
        print(f"- avg_entropy_bits        : {totals['avg_entropy_bits']:.4f}")
        if totals["avg_norm_entropy"] is not None:
            print(f"- avg_norm_entropy        : {totals['avg_norm_entropy']:.4f}")
        print(f"- avg_perplexity          : {totals['avg_perplexity']:.2f}")

    return {"per_layer": results, "summary": totals}


def make_prefix(num_layers: int):
    """根据层数生成形如 <a_{}>, <b_{}> ... 的前缀列表"""
    base = []
    for i in range(num_layers):
        ch = chr(ord('a') + (i % 26))
        suffix = "" if i < 26 else str(i // 26)
        base.append(f"<{ch}{suffix}_{{}}>")
    return base


def main():
    # ======== 配置（自行修改） ========
    # dataset = "Beauty"
    dataset = "Fuse"
    ckpt_path = f"./ckpt/Fuse_all_split/Oct-19-2025_16-25-58/best_collision_model.pth"
    output_file = f"../../Our_idea/data/Fuse/{dataset}_t5_rqvae_super_code_3072.npy"
    device = torch.device("cuda:7")  # 如需CPU可改为 torch.device("cpu")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ======== 加载模型与数据 ========
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    print("data_path:", args.data_path)
    data = EmbDataset(args.data_path)
    print("Dataset length:", len(data))  # 期望值 N

    use_post_linear = getattr(args, "use_post_linear", False)
    post_linear_bias = getattr(args, "post_linear_bias", True)
    model = RQVAE(
        in_dim=data.dim,
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
        use_post_linear=use_post_linear,
        post_linear_bias=post_linear_bias,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
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
