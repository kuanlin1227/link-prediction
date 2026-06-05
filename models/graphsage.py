"""
models/graphsage.py
GraphSAGE link prediction 模型

架構：
  encoder: 2 層 SAGEConv → 節點 embedding z
  decoder: z[u] · z[v]（內積）→ 連結機率
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, cohen_kappa_score,
)


class LinkSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        """內積解碼器：分數越高表示越可能連結"""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def train_epoch(model, optimizer, train_data, device) -> float:
    model.train()
    optimizer.zero_grad()

    x           = train_data.x.to(device)
    edge_index  = train_data.edge_index.to(device)
    eli         = train_data.edge_label_index.to(device)
    labels      = train_data.edge_label.float().to(device)

    pred = model(x, edge_index, eli)
    loss = F.binary_cross_entropy_with_logits(pred, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, train_data, eval_data, device, threshold: float = 0.5) -> dict:
    """
    使用訓練集的 edge_index 做 message passing，
    但在 eval_data 的邊上計算各項指標。

    AUC / AP 不需 threshold（用連續分數計算）；
    混淆矩陣 (TN/FP/FN/TP) 與其衍生指標 (accuracy/precision/recall/f1)
    則以 sigmoid 機率 >= threshold（預設 0.5）判為「有連結」。
    """
    model.eval()

    x           = train_data.x.to(device)
    edge_index  = train_data.edge_index.to(device)
    eli         = eval_data.edge_label_index.to(device)
    labels      = eval_data.edge_label.cpu().numpy()

    z    = model.encode(x, edge_index)
    pred = model.decode(z, eli).sigmoid().cpu().numpy()

    # 以 threshold 二值化後算混淆矩陣（labels=[0,1] 確保固定為 2x2）
    pred_label = (pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_label, labels=[0, 1]).ravel()

    eps       = 1e-12
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    kappa     = cohen_kappa_score(labels, pred_label)

    return {
        'auc':       roc_auc_score(labels, pred),
        'ap':        average_precision_score(labels, pred),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'kappa':     kappa,
    }
