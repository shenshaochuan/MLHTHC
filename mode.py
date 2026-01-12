import os
import random
import pickle
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from dhg.nn import HGNNConv
from dhg import Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt

# ------------------- 配置 -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_PKL = r""
INTERACTION_CSV = r""
OUTPUT_MODEL = ""
os.makedirs("curve_data", exist_ok=True)

# ------------------- 加载超图 -------------------
with open(DATA_PKL, "rb") as f:
    data = pickle.load(f)
    hypergraph = data["combined_hypergraph"]

node_to_index = {node: idx for idx, node in enumerate(hypergraph.nodes)}
num_nodes = len(hypergraph.nodes)
print(f"num_nodes = {num_nodes}")

e_list = []
e_weights = []
for edge_id in hypergraph.edges:
    nodes = [node_to_index[node] for node in hypergraph.edges[edge_id] if node in node_to_index]
    if len(nodes) >= 2:
        e_list.append(nodes)

dhg_hypergraph = Hypergraph(num_nodes, e_list, e_weights)

# ------------------- 初始化特征 -------------------
node_features = torch.randn((num_nodes, 128))


# ------------------- 模型 -------------------
class HypergraphTransformerConv(nn.Module):
    def __init__(self, in_channels, hgnn_hidden, transformer_hidden,
                 nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(HypergraphTransformerConv, self).__init__()
        self.hgnn1 = HGNNConv(in_channels, transformer_hidden)
        self.hgnn_norm = nn.LayerNorm(transformer_hidden)

        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_hidden,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_proj = nn.Linear(transformer_hidden, transformer_hidden)
        self.activation = nn.GELU()

    def forward(self, x, hg):
        x = self.hgnn1(x, hg)
        x = F.relu(x)
        x = self.hgnn_norm(x)

        x = x.unsqueeze(0)  # [1, N, D]
        x = self.transformer(x)
        x = x.squeeze(0)  # [N, D]

        x = self.final_proj(x)
        x = self.activation(x)
        return x  # [num_nodes, D]


class MLPSimilarity(nn.Module):
    def __init__(self, input_dim):
        super(MLPSimilarity, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2, torch.abs(x1 - x2), x1 * x2], dim=-1)
        score = self.mlp(combined)
        return score.view(-1)


# ------------------- 构建正负样本 -------------------
interaction_df = pd.read_csv(INTERACTION_CSV)
all_pos_pairs = []
for _, row in interaction_df.iterrows():
    a = row['drug_id']
    b = row['interaction_drug_id']
    if a in node_to_index and b in node_to_index:
        ia = node_to_index[a]
        ib = node_to_index[b]
        pair = (min(ia, ib), max(ia, ib))
        all_pos_pairs.append(pair)
all_pos_pairs = list(set(all_pos_pairs))
print(f"num pos pairs = {len(all_pos_pairs)}")


def generate_neg_samples(num_neg, existing_set, num_nodes):
    neg_pairs = []
    trials = 0
    while len(neg_pairs) < num_neg and trials < num_neg * 10:
        a, b = random.sample(range(num_nodes), 2)
        pair = (min(a, b), max(a, b))
        if pair not in existing_set:
            neg_pairs.append(pair)
            existing_set.add(pair)
        trials += 1
    return neg_pairs


# 固定一批负样本（1*正样本）
existing_set = set(all_pos_pairs)
fixed_neg_pairs = generate_neg_samples(len(all_pos_pairs) * 1, existing_set.copy(), num_nodes)
print(f"num fixed neg pairs = {len(fixed_neg_pairs)}")

# 合并并打乱，然后划分训练/测试/验证（80/10/10）
pairs_list = [[a, b, 1] for (a, b) in all_pos_pairs] + [[a, b, 0] for (a, b) in fixed_neg_pairs]
pairs = torch.tensor(pairs_list, dtype=torch.long)
perm = torch.randperm(len(pairs))
pairs = pairs[perm]

# 计算划分点
total = len(pairs)
train_split = int(0.8 * total)  # 80% 训练集
remaining = total - train_split
test_split = int(0.5 * remaining)  # 剩余部分的一半作为测试集（总数据的10%）

# 划分数据集
train_pairs = pairs[:train_split].clone()
test_pairs = pairs[train_split:train_split + test_split].clone()
val_pairs = pairs[train_split + test_split:].clone()

print(f"train samples = {len(train_pairs)}, test samples = {len(test_pairs)}, val samples = {len(val_pairs)}")
print(f"划分比例: {len(train_pairs) / total:.2f}:{len(test_pairs) / total:.2f}:{len(val_pairs) / total:.2f}")

# ------------------- 训练准备 -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")
try:
    dhg_hypergraph = dhg_hypergraph.to(device)
except Exception:
    pass

model = HypergraphTransformerConv(128, 128, 256, nhead=4, num_layers=3, dim_feedforward=512).to(device)
mlp_model = MLPSimilarity(256).to(device)
node_features = node_features.to(device)

optimizer = optim.Adam(list(model.parameters()) + list(mlp_model.parameters()), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
bce_loss_fn = nn.BCEWithLogitsLoss()
rank_loss_fn = nn.MarginRankingLoss(margin=1.0)

num_epochs = 50
batch_size = 512
best_auc = 0.0
best_state = None
patience = 10
wait = 0

# ------------------- 训练循环 -------------------
for epoch in range(num_epochs):
    model.train()
    mlp_model.train()

    # 对训练样本打乱
    perm = torch.randperm(len(train_pairs))
    train_shuffled = train_pairs[perm]

    epoch_loss_sum = 0.0
    total_samples = 0

    # 梯度累积控制
    grad_accum_steps = 4  # 你可调小以降低 GPU 内存占用
    accum_loss = 0.0
    accum_count = 0

    for i in range(0, len(train_shuffled), batch_size):
        batch = train_shuffled[i:i + batch_size].to(device)
        a_idx = batch[:, 0]
        b_idx = batch[:, 1]
        labels = batch[:, 2].float().to(device)

        # 前向计算
        embeddings = model(node_features, dhg_hypergraph)  # [num_nodes, dim]

        x1 = embeddings[a_idx]
        x2 = embeddings[b_idx]
        logits = mlp_model(x1, x2)  # [B]

        # BCE loss
        bce = bce_loss_fn(logits, labels)

        # Ranking loss
        pos_scores = logits[labels == 1]
        neg_scores = logits[labels == 0]
        if pos_scores.numel() > 0 and neg_scores.numel() > 0:
            k = min(min(pos_scores.numel(), neg_scores.numel()), 32)
            pos_idx = torch.randperm(pos_scores.numel(), device=device)[:k]
            neg_idx = torch.randperm(neg_scores.numel(), device=device)[:k]
            pos_sample = pos_scores[pos_idx]
            neg_sample = neg_scores[neg_idx]
            rank_loss = rank_loss_fn(pos_sample, neg_sample, torch.ones_like(pos_sample, device=device))
        else:
            rank_loss = torch.tensor(0.0, device=device)

        loss = bce + 0.1 * rank_loss

        # 累积损失
        accum_loss += loss
        accum_count += 1
        epoch_loss_sum += loss.item() * len(batch)
        total_samples += len(batch)

        # 梯度更新
        do_step = (accum_count % grad_accum_steps == 0) or (i + batch_size >= len(train_shuffled))
        if do_step:
            optimizer.zero_grad()
            accum_loss.backward()
            optimizer.step()
            accum_loss = 0.0
            accum_count = 0

    scheduler.step()
    epoch_loss = epoch_loss_sum / (total_samples + 1e-12)

    # ---------------- 验证 ----------------
    model.eval()
    mlp_model.eval()
    with torch.no_grad():
        embeddings_val = model(node_features, dhg_hypergraph)
        a_val = val_pairs[:, 0].to(device)
        b_val = val_pairs[:, 1].to(device)
        labels_val = val_pairs[:, 2].float().to(device)

        logits_val = mlp_model(embeddings_val[a_val], embeddings_val[b_val])
        probs_val = torch.sigmoid(logits_val).cpu().numpy()
        labels_val_np = labels_val.cpu().numpy()

        try:
            auc = roc_auc_score(labels_val_np, probs_val)
            aupr = average_precision_score(labels_val_np, probs_val)
            preds = (probs_val >= 0.7).astype(int)
            f1 = f1_score(labels_val_np, preds)
        except Exception:
            auc, aupr, f1 = 0.0, 0.0, 0.0

    # 保存最佳模型
    if auc > best_auc:
        best_auc = auc
        best_state = {"model": model.state_dict(), "mlp": mlp_model.state_dict()}
        wait = 0
    else:
        wait += 1

    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch {epoch + 1:03d} | Loss {epoch_loss:.6f} | Val AUC {auc:.4f} | AUPR {aupr:.4f} | F1 {f1:.4f} | Best {best_auc:.4f} | Wait {wait} | LR {current_lr:.2e}")

    if wait >= patience:
        print(f"Early stopping triggered. Best AUC={best_auc:.4f}")
        break

# ------------------- 保存最佳模型 -------------------
if best_state is not None:
    torch.save(best_state, OUTPUT_MODEL)
    print(f"Saved best model to {OUTPUT_MODEL} (AUC={best_auc:.4f})")
else:
    print("No improvement observed; no model saved.")

# ------------------- 最终评估与绘图（在测试集上） -------------------
if best_state is not None:
    model.load_state_dict(best_state["model"])
    mlp_model.load_state_dict(best_state["mlp"])
    model.eval()
    mlp_model.eval()
    with torch.no_grad():
        # 在测试集上评估
        embeddings_test = model(node_features, dhg_hypergraph)
        a_test = test_pairs[:, 0].to(device)
        b_test = test_pairs[:, 1].to(device)
        labels_test = test_pairs[:, 2].float().to(device)

        logits_test = mlp_model(embeddings_test[a_test], embeddings_test[b_test])
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        labels_test_np = labels_test.cpu().numpy()

        try:
            test_auc = roc_auc_score(labels_test_np, probs_test)
            test_aupr = average_precision_score(labels_test_np, probs_test)
            test_preds = (probs_test >= 0.5).astype(int)
            test_f1 = f1_score(labels_test_np, test_preds)
            print(f"Test set performance - AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, F1: {test_f1:.4f}")

            fpr, tpr, _ = roc_curve(labels_test_np, probs_test)
            precision, recall, _ = precision_recall_curve(labels_test_np, probs_test)
        except Exception:
            fpr = tpr = precision = recall = np.array([])
            test_auc = 0.0

    # 保存曲线数据
    with open("curve_data/transformer_roc_curve_data_811.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FPR", "TPR"])
        for fp, tp in zip(fpr, tpr):
            writer.writerow([fp, tp])

    with open("curve_data/transformer_pr_curve_data_811.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Recall", "Precision"])
        for r, p in zip(recall, precision):
            writer.writerow([r, p])

    # 保存性能指标图
    plt.figure(figsize=(5, 3))
    plt.text(0.1, 0.6, f"Test AUC  = {test_auc:.4f}", fontsize=12)
    plt.text(0.1, 0.4, f"Epochs    = {epoch + 1}", fontsize=12)
    plt.axis('off')
    plt.title("Best Performance Metrics (Test Set)")
    plt.tight_layout()
    plt.savefig("best_metrics_transformer_811.png")
    plt.close()

    # 绘制ROC曲线
    if fpr.size and tpr.size:
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test Set)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("transformer_best_auc_811.png")
        plt.close()

    # 绘制PR曲线
    if recall.size and precision.size:
        plt.figure()
        plt.plot(recall, precision, label=f"AUPR = {test_aupr:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Test Set)")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("transformer_best_aupr_811.png")
        plt.close()

print("Done.")
