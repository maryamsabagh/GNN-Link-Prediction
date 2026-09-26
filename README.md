# GNN Link Prediction

Link prediction and recommendation on bipartite graphs with graph neural networks, tested on two real-world datasets:

- **Spotify:** a playlist–track graph built from the Spotify Million Playlist Dataset.
- **Amazon:** a user–item graph from the Amazon-Book data used in the LightGCN paper.

This project studies message passing neural networks (MPNNs), a family of GNNs that works well for recommendation and link prediction. It combines structural analysis of the graphs with experiments. Starting from a LightGCN-style model, we swap in different message-passing layers (LightGCN, GraphSAGE, and Chebyshev convolutions) and train each one with Bayesian Personalized Ranking (BPR) loss and controlled negative sampling. The task is to recover hidden edges between the two node types. Every model runs on the same data splits, so the results can be compared directly.

## How it works

1. **Load the data.** For Spotify, read the playlist JSON files in `data/` into `Playlist` and `Track` objects. For Amazon, read the user–item lists in `amazon_example/amazon_data/`.
2. **Build a bipartite graph.** Playlists (or users) are one type of node and tracks (or items) are the other. An edge means "this track is in this playlist" (or "this user interacted with this item").
3. **Keep the dense core.** Take the k-core of the graph by repeatedly removing nodes with fewer than k connections. Spotify uses k = 30 and Amazon uses k = 26. This helps filter out noise.
4. **Optionally add artists (Spotify only).** Tracks can also be linked to their artists, which gives the model extra structure to pass messages through.
5. **Split the edges** into train (70%), validation (15%) and test (15%) sets. The model has to recover the held-out edges.
6. **Train a GNN** that learns an embedding vector for every node. A pair's score is the dot product of the two nodes' embeddings.
7. **Evaluate** with two metrics:
   - ROC-AUC: how well real edges are separated from sampled non-edges.
   - Recall@K: how many hidden edges show up in each node's top recommendations.

## Project structure

| Path | Purpose |
|---|---|
| `main.py` | Spotify experiment. Loads the data, builds the 30-core graph, splits it, and trains the model over 5 runs. |
| `amazon_main.py` | Amazon experiment. Same pipeline on the 26-core user–item graph. |
| `basic_types.py` | Classes for the raw Spotify data: `Track`, `Playlist`, `JSONFile`. |
| `spotify_data_loader.py` | Reusable function that loads the first N Spotify data files. |
| `networkx_load_data.py` | Loads the Amazon `train.txt` into a NetworkX graph and takes its k-core. |
| `graph_building_routines.py` | Builds graphs, relabels nodes as integers, adds artist edges, and creates the edge splits. |
| `gcn_class.py` | The `GCN` model: node embeddings, message-passing layers, scoring and losses. |
| `BPR_class.py` | The `BPRLoss` loss function. |
| `sampling_methods.py` | Random and hard negative edge sampling. |
| `train_and_test.py` | Training loop, evaluation function and ROC-AUC metric. |
| `recall_measurement.py` | Recall@K evaluation. |
| `graph_analyzer.py` | Helpers for degree statistics and plots. |
| `data/` | Spotify Million Playlist Dataset slices (`mpd.slice.*.json`). |
| `30core_first_30.pkl`, `amazon_26core.pkl` | Cached k-core graphs, so they don't have to be rebuilt every run. |
| `amazon_example/` | The Amazon dataset plus a reference copy of the original LightGCN implementation (`Light_GCN_Git_Clone/`). |


## Main functions and classes

### Data loading

- **`Track`**: one track, with its URI, name, artist URI, artist name, and the playlist it belongs to.
- **`Playlist`**: one playlist. `load_tracks()` fills it with its `Track` objects. Playlists are named `playlist_<index>`.
- **`JSONFile`**: loads one data file. `process_file()` turns each playlist in the file into a `Playlist`, and numbering continues where the previous file stopped, so playlist names never collide.
- **`spotify_data_loader(N_FILES_TO_USE, DATA_DIR)`**: loads the first N Spotify files. Returns the playlists, tracks and artists, plus the playlist–track and track–artist edge lists.
- **`networkx_load_data.Data(path, kcore)`**: reads the Amazon training file (one line per user: `user_id item_id item_id ...`), builds the user–item graph, and keeps its k-core in `.G`.

### Graph construction: `graph_building_routines.py`

- **`graph_attribute_builder(nodes1, attribute1, nodes2, attribute2, edges)`**: builds a NetworkX graph with two node types, each tagged with a `node_type`.
- **`playlist_track_graph(...)`**: builds the playlist–track graph and takes its k-core.
- **`extra_attribute_edge_index(...)`**: builds track–artist edges for the tracks that survived the k-core. Artist nodes are numbered after the tracks. Returns the number of artists and an edge index tensor.
- **`graph_relabel_and_sort(G)`**: replaces node names with integer ids (`node2id`, `id2node`). Sorting puts every playlist before every track, so ids `0 … num_playlists-1` are playlists and the rest are tracks.
- **`train_valid_test_split(G)`**: splits the edges 70/15/15 with PyG's `RandomLinkSplit`. Each split has an `edge_index` (the edges used for message passing) and an `edge_label_index` (the edges the model has to predict).

### The model: `gcn_class.py`

**`GCN(num_nodes, embedding_dim, num_layers, conv_layer="LGC")`** is adapted from PyTorch Geometric's LightGCN. It learns one embedding vector per node and refines it through a stack of graph convolution layers. The final embedding is a weighted sum of the outputs of every layer, including the starting embedding. The weights are equal by default, and `alpha_learnable=True` makes them learnable.

`conv_layer` picks the message-passing layer:

| Value | Layer |
|---|---|
| `"LGC"` | LightGCN convolution (`LGConv`) |
| `"SAGE"` | GraphSAGE (`SAGEConv`) |
| `"CHEB"` | Chebyshev spectral convolution (`ChebConv`, K = 3) |


Key methods:

- **`get_embedding(edge_index)`**: runs message passing and returns every node's final embedding.
- **`predict_link_embedding(embed, edge_label_index)`**: scores each pair as the dot product of its embeddings.
- **`recommend(edge_index, src_index, dst_index, k)`**: returns the top-k highest-scoring destination nodes for each source node.
- **`recommendation_loss(pos, neg, num_neg_edges, lambda_reg)`**: BPR loss via `BPRLoss`.

### Loss: `BPR_class.py`

**`BPRLoss`** implements Bayesian Personalized Ranking. Each real edge is compared with `num_neg_edges` sampled non-edges, and the loss rewards the model when the real edge scores higher. The result is averaged over all pairs, with optional L2 regularization on the embeddings.

### Negative sampling: `sampling_methods.py`

Training needs examples of "no edge" to contrast with the real ones.

- **`sample_negative_edges(...)`** (default): for every real edge, picks `num_neg_edges` tracks the playlist is *not* connected to.
- **`sample_hard_negative_edges(...)`**: scores every track for each playlist with the current model, ignores real edges, and picks a negative from the highest-scoring tracks (the ones the model wrongly likes most). The candidate pool shrinks from 100% to 50% of tracks over training, so the negatives get harder over time.
- **`sample_negative_edges_fast(...)`** / **`sample_negative_edges_nocheck(...)`**: faster variants. The `nocheck` version doesn't verify that the pairs it picks are really non-edges.

### Training and evaluation: `train_and_test.py`

- **`train(datasets, model, optimizer, loss_fn, args, ...)`**: the training loop. Each epoch it:
  1. samples negatives,
  2. computes embeddings from the training graph,
  3. scores real and negative edges,
  4. computes the loss (`"BPR"` or `"BCE"`) and updates the model,
  5. evaluates on the validation set and prints train and validation loss and ROC-AUC.

 
- **`test(model, data, ...)`**: evaluates the model on a split without updating it and returns the loss and ROC-AUC.
- **`metrics(labels, preds)`**: ROC-AUC, the probability that a real edge scores higher than a negative one.

### Recall: `recall_measurement.py`

**`recall_at_k(data, model, k, ...)`**: for each playlist, it:
1. scores every track,
2. excludes edges already used for message passing,
3. keeps the top k,
4. counts how many hidden edges are among them.

Recall is the fraction of hidden edges found, averaged over playlists.

### Analysis helpers: `graph_analyzer.py`

- **`degree_sequence(G)`**: node degrees, sorted from highest to lowest.
- **`plt_node_rank(deg_seq)`**: degree rank plot.
- **`plt_deg_distribution(deg_seq, nbins)`**: log–log degree distribution.

## Running it

### Requirements

Python 3.9+ with:

```
torch
torch_geometric
networkx
numpy
scipy
scikit-learn
matplotlib
tqdm
```

### Default settings

Both scripts use:
- 3 layers and 64-dimensional embeddings
- learning rate 0.01, weight decay 1e-5
- 301 epochs
- BPR loss, 10 random negatives per positive edge
- 5 independent runs

To change the model, edit `conv_layer` in the script. To change the sampler, edit `neg_samp` (`"random"` or `"hard"`).

### Steps

1. Set `MAIN_DIR` and `DATA_PATH` at the top of `main.py` (or `amazon_main.py`) to your local paths. They currently point to `/home/jovyan/...`.
2. Run an experiment:

   ```bash
   python main.py         # Spotify
   python amazon_main.py  # Amazon
   ```

3. Explore the saved results with the notebooks in `spotify_analysis/` and `amazon_analysis/`.



## Acknowledgements

- The model is adapted from the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) LightGCN implementation.
- The Amazon data loader is based on the [NGCF](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) / [LightGCN](https://github.com/kuandeng/LightGCN) code by Xiang Wang et al.
- Spotify data: [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).
