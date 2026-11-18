# Install and data versions

The repository holds two distributions. `fdia-graph` generates a timeline and loads it.
`fdia-analysis` runs estimators and localizers on one, and installs `fdia-graph` with it.

## The generation toolkit

| Command | Gives |
|---|---|
| `pip install fdia-graph` | the loader: opens a timeline and exports whole splits as numpy arrays |
| `pip install "fdia-graph[torch]"` | adds PyTorch: single records, batches and the DataLoader |
| `pip install "fdia-graph[pyg]"` | adds torch_geometric records and timelines |
| `pip install "fdia-graph[generate]"` | adds pandapower, to generate timelines |
| `pip install "fdia-graph[iso]"` | adds gridstatus, to download CAISO and ERCOT load profiles |
| `pip install "fdia-graph[ny]"` | adds what the New York build needs: pandapower and Pyomo |
| `pip install "fdia-graph[all]"` | every extra above |

NYISO load profiles need no extra. The New York build also needs Ipopt, a solver that pip does
not install: see [Solving](../concepts/solving.md) and
[Build the operating points](operating_points.md).

## The analysis toolkit

| Command | Gives |
|---|---|
| `pip install fdia-analysis` | the toolkit and `fdia-graph`: the threshold localizers and the results store |
| `pip install "fdia-analysis[se]"` | adds pandapower and scipy, for `fdia_analysis.se` and the residual localizer |
| `pip install "fdia-analysis[torch]"` | adds torch, for the learned localizers |
| `pip install "fdia-analysis[federated]"` | adds torch and scikit-learn, for `fdia_analysis.federated` |
| `pip install "fdia-analysis[all]"` | every analysis extra above |

## Data versions

A released dataset belongs to a data release. Each package version pins one data release, so
the same code always reads the same data.

- `fg.load("ieee118")` reads the pinned release, or the one `FDIA_GRAPH_RELEASE` names when
  that variable is set.
- `fg.load("ieee118", release="v0.8.1")` reads another one. Any timeline release from v0.8.0 can
  be read.
- `pip install --upgrade fdia-graph` moves the pin forward when the new version pins a newer
  release.

## The cache

A downloaded file is kept in `~/.cache/fdia_graph` and is downloaded once. Set
`FDIA_GRAPH_CACHE` to use another folder. After a download the package checks the file's sha256
for the releases v0.8.0, v0.8.1 and v0.8.3; a file of another release is not checked.
