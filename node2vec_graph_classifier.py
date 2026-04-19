import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import Node2Vec
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import umap

# ── 資料載入 ───────────────────────────────────────────────
dataset = KarateClub()
data    = dataset[0]
print(data)
print(f"節點數: {data.num_nodes}  邊數: {data.num_edges // 2}  類別數: {data.y.max().item() + 1}\n")

# ── Node2Vec 模型 ──────────────────────────────────────────
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
)

loader    = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ── 訓練 ──────────────────────────────────────────────────
print("訓練 Node2Vec...")
epoch_losses = []
model.train()
for epoch in range(100):
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    epoch_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1:3d} / 100  Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "node2vec_model.pt")
print("模型已儲存至 node2vec_model.pt\n")

# ── 取得嵌入 ───────────────────────────────────────────────
model.eval()
z    = model().detach().cpu().numpy()
y_np = data.y.numpy()

# ── 節點分類 ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    z, y_np, test_size=0.2, random_state=42
)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc    = clf.score(X_test, y_test)

print(f"節點分類準確率：{acc:.4f}\n")
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

results_df = pd.DataFrame({"node": range(len(y_np)), "true_label": y_np, "embedding_norm": np.linalg.norm(z, axis=1)})
results_df.to_csv("node_embeddings_meta.csv", index=False)

# ── 降維（t-SNE vs UMAP）─────────────────────────────────
z_tsne = TSNE(n_components=2, random_state=42).fit_transform(z)
z_umap = umap.UMAP(n_neighbors=10, random_state=42).fit_transform(z)

# ── 視覺化（2×2）─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 左上：Loss Curve
axes[0, 0].plot(range(1, 101), epoch_losses, color="steelblue", linewidth=1.5)
axes[0, 0].set_title("Training Loss Curve")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

# 右上：Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(ax=axes[0, 1], colorbar=False)
axes[0, 1].set_title(f"Confusion Matrix  (Acc = {acc:.2f})")

# 左下：t-SNE
sc1 = axes[1, 0].scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_np, cmap="tab10", s=100, edgecolors="white", linewidths=0.5)
for i, (x, y) in enumerate(z_tsne):
    axes[1, 0].annotate(str(i), (x, y), fontsize=6, ha="center", va="center")
axes[1, 0].set_title("t-SNE of Node2Vec Embeddings")
axes[1, 0].axis("off")
plt.colorbar(sc1, ax=axes[1, 0], label="Community")

# 右下：UMAP
sc2 = axes[1, 1].scatter(z_umap[:, 0], z_umap[:, 1], c=y_np, cmap="tab10", s=100, edgecolors="white", linewidths=0.5)
for i, (x, y) in enumerate(z_umap):
    axes[1, 1].annotate(str(i), (x, y), fontsize=6, ha="center", va="center")
axes[1, 1].set_title("UMAP of Node2Vec Embeddings")
axes[1, 1].axis("off")
plt.colorbar(sc2, ax=axes[1, 1], label="Community")

plt.suptitle("Node2Vec on Karate Club — Full Analysis", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("node2vec_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n已儲存 node2vec_analysis.png、node_embeddings_meta.csv、node2vec_model.pt")
