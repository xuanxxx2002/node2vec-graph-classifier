# node2vec-graph-classifier

使用 Node2Vec 對 Karate Club 圖形學習節點嵌入，結合 t-SNE / UMAP 降維視覺化與 Logistic Regression 節點分類，完整呈現 Graph Embedding 的訓練與評估流程。

## 分析流程

```
KarateClub Graph
      ↓
Node2Vec 訓練 (128-dim embeddings, 100 epochs)
      ↓
┌─────────────────────┬──────────────────────┐
節點分類                    降維視覺化
Logistic Regression     t-SNE  vs  UMAP
Accuracy / F1 / CM
```

## 輸出

| 檔案 | 內容 |
|---|---|
| `node2vec_analysis.png` | 2×2 圖表：Loss Curve、Confusion Matrix、t-SNE、UMAP |
| `node_embeddings_meta.csv` | 每個節點的真實標籤與 Embedding Norm |
| `node2vec_model.pt` | 訓練完成的 Node2Vec 模型權重 |

## 視覺化說明

| 圖表 | 說明 |
|---|---|
| Loss Curve | 100 個 Epoch 的訓練損失變化 |
| Confusion Matrix | 節點分類結果（含準確率）|
| t-SNE | 128 維嵌入降至 2D（t-SNE 方法）|
| UMAP | 128 維嵌入降至 2D（UMAP 方法，速度更快）|

## 快速開始

**安裝依賴**

```bash
python -m pip install torch torch-geometric scikit-learn umap-learn matplotlib pandas
```

**Step 2 — 安裝 torch-cluster（Node2Vec 必要）**

先確認 PyTorch 版本：
```bash
python -c "import torch; print(torch.__version__)"
```

再依版本安裝（以 `2.x.x+cpu` 為例）：
```bash
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-{版本號}.html
```

例如 PyTorch `2.11.0+cpu`：
```bash
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

> 完整 wheel 清單：https://data.pyg.org/whl/

**執行**

```bash
python node2vec_graph_classifier.py
```

## 資料集

[Zachary's Karate Club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club)（34 節點、78 邊、4 個社群），由 PyTorch Geometric 內建提供。
