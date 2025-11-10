"""
# Running CellOracle on the Reference Atlas (Rosshandler et al., 2024)

The goal of this analysis is to evaluate whether bulk RNA-seq differentially expressed genes (DEGs) 
are expressed along the neuromesodermal progenitor (NMP) trajectory in the Rosshandler et al. reference atlas. 
By doing so, we can prioritise genes for downstream validation and mechanistic investigation.

The dataset is already preprocessed and contains sufficient cell numbers for this analysis.

## Objectives
1. Identify DEGs from bulk RNA-seq that intersect with genes expressed in the reference atlas.
2. Subset the reference atlas to include only the NMP trajectory and its derivatives.
3. Visualise the normalised expression of intersecting genes within the atlas.
4. (Optional) Generate metacells to mitigate sparsity in single-cell data.
5. Run CellOracle’s perturbation model using the default mouse gene regulatory network (GRN) 
   for the selected genes.

Note: CellOracle version must be >= v0.10.10.

## Implementation Notes
- Column cleanup:
  Certain columns (e.g. 'Highly variable') should be removed to prevent conflicts during downstream
  subsetting and plotting.

- UMAP and clustering:
  Due to instability in the PAGA graph post-subsetting, this analysis will rely on the original UMAP
  embeddings. Clustering in the subsetted object showed inconsistencies, suggesting that reannotation
  or further cleaning may be required.

- Precomputed data:
  Most analyses will use the existing precomputed information from the atlas. Reprocessing may be
  considered later to confirm reproducibility and stability of results.

- Dimensionality reduction:
  Principal components (PCs) in the original object were batch-corrected using BBKNN, which replaces
  `sc.neighbors`. The existing neighbor graph is suitable for diffusion map creation for pseudotime
  estimation.

## Environment Notes
- Python environment: 3.8
- Reason: Compatibility issues with fa2 encountered under Python 3.9.
"""

# Import packages
import os
from re import search
from dfply import *
import sys
from matplotlib import pyplot as plt
from scipy.sparse import issparse
from scipy import sparse
import anndata as ann
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import importlib
import datetime
from pathlib import Path
import pandas as pd
from scipy.sparse import issparse

sc.settings.verbosity = 3
sc.settings.set_figure_params(
    dpi=150,
    facecolor="white",
    # color_map="YlGnBu",
    frameon=False,
)

# visualization settings
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

plt.rcParams['figure.figsize'] = [9, 7]
plt.rcParams["savefig.dpi"] = 400

import celloracle as co
co.check_python_requirements()
co.__version__

"""
/home/test/.local/lib/python3.8/site-packages/loompy/bus_file.py:68: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  def twobit_to_dna(twobit: int, size: int) -> str:
/home/test/.local/lib/python3.8/site-packages/loompy/bus_file.py:85: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  def dna_to_twobit(dna: str) -> int:
/home/test/.local/lib/python3.8/site-packages/loompy/bus_file.py:102: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  def twobit_1hamming(twobit: int, size: int) -> List[int]:
'0.16.0'
"""

save_folder = "/data/kellyrc/Testing_Tools/CellOracle"
os.makedirs(save_folder, exist_ok=True)

# Add in day and date to record when things were last performed. 
today = datetime.date.today().strftime('%y%m%d')
today

os.getcwd()

# Get the current date and time
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))


# Load in the data from the reference project and go from there.
reference_data = sc.read_h5ad(   
    filename="/data/kellyrc/Testing_Tools/reference_data/Rosshandler_extended_atlas.h5ad"
)
# 430339 cells across all stages
# We dont need all stages/cell types.
# Plot and review the data annotations
sc.pl.umap(
    reference_data,
    color=["celltype_extended_atlas"],
    legend_loc="right margin",
)
# Remove certain columns to avoid issues downstream: Not sure why but i think they were being overwritten?
remove_obs = ['louvain','leiden']
remove_var = ['highly_variable','dispersions','dispersions_norm']
reference_data.obs.drop(columns=remove_obs, inplace=True)
reference_data.var.drop(columns=remove_var, inplace=True)
reference_data

# Reorganise the object.
reference_data.X = sparse.csr_matrix(reference_data.X)
reference_data.layers["raw_counts"] = reference_data.raw.X.copy()
reference_data.layers["Log_norm_original"] = reference_data.X.copy() # we will renorm so call "original"

# Convert the var.index to the mgi_symbol NOT Ensembl IDs. 
reference_data.var["ensembl_id"] = reference_data.var.index
reference_data.var.index = reference_data.var["mgi_symbol"]
reference_data.var.index = reference_data.var.index.tolist() # converts to categorical index
reference_data.var.index

reference_subset = reference_data[reference_data.obs.stage.isin(['E8.5'])]
reference_subset.X = reference_subset.layers['raw_counts']
reference_subset #74007 cells in total.
del(reference_data)

# Look at the options of cell tyoes
celltypes = reference_subset.obs['celltype_extended_atlas']
# Create a table of unique options and their counts
celltype_table = pd.DataFrame(celltypes.value_counts()).reset_index()
celltype_table.columns = ['Cell Type', 'Count']

# If the data is in dataframe format. We can just make a simple table to review the data. 
pd.set_option('display.max_rows', None) # want to plot the entire table.
print(celltype_table)

"""
                                         Cell Type  Count
0                                        Erythroid   8361
1                                        Allantois   2814
2                    Ventral hindbrain progenitors   2006
3                                    Optic vesicle   1972
4                          Spinal cord progenitors   1871
5                   Dorsal spinal cord progenitors   1849
6                                     ExE endoderm   1792
7                      Midbrain/Hindbrain boundary   1767
8                           Lateral plate mesoderm   1762
9                           Migratory neural crest   1729
10                                      Mesenchyme   1700
11                                            NMPs   1666
12                             Presomitic mesoderm   1343
13                                      Sclerotome   1323
14                             Pharyngeal endoderm   1207
15                                   Limb mesoderm   1074
16                                Somitic mesoderm   1047
17                           Allantois endothelium    967
18                    Dorsal hindbrain progenitors    912
19                               Kidney primordium    889
20                                   Limb ectoderm    872
21                            NMPs/Mesoderm-biased    862
22                                  YS mesothelium    826
23                                        Endotome    719
...
78                               Parietal endoderm      2
79                         Otic neural progenitors      2
80                           Intermediate mesoderm      2
81                          Frontonasal mesenchyme      1
"""

# Lets collect the relevant celltypes
reference_trajectory = reference_subset[reference_subset.obs.celltype_extended_atlas.isin(['NMPs','NMPs/Mesoderm-biased',
                                                                                           'Somitic mesoderm',
                                                                                           'Posterior somitic tissues',
                                                                                           'Spinal cord progenitors',
                                                                                           'Dorsal spinal cord progenitors',
                                                                                          'Presomitic mesoderm','Sclerotome',
                                                                                          'Lateral plate mesoderm','Caudal mesoderm']),:]
# Look at the options of cell tyoes
celltypes = reference_trajectory.obs['celltype_extended_atlas']
# Create a table of unique options and their counts
celltype_table = pd.DataFrame(celltypes.value_counts()).reset_index()
celltype_table.columns = ['Cell Type', 'Count']

# Assuming celltype_table is your DataFrame
pd.set_option('display.max_rows', None) # want to plot the entire table.
print(celltype_table)

# Based on the table, i've decided to remove the anterior primitive streak, 'Caudal epiblast' and 'Caudal mesoderm'. Just not enough cells. 
# Now we have 9 cell types. Lets plot the UMAP using their original coordinates. 

sc.pl.draw_graph(
    reference_trajectory,
    color=["celltype_extended_atlas"],
    legend_loc="right margin"
)
# this looks decent, the NMP and NMP/Mesobias cells are a little bit off so maybe renormalising and plotting can fix this. 
# Ideally, with time we can simply reprocess everything but up until this point, we are looking good. 


"""
# Renormalise the data and replot everything.

Given that i've subsetted the data, I should  renormalise and plot. For this, i'll run through the CellOracle scRNAseq preprocessing with scanpy (https://morris-lab.github.io/CellOracle.documentation/notebooks/03_scRNA-seq_data_preprocessing/scanpy_preprocessing_with_Paul_etal_2015_data.html#7.-Cell-clustering).
I have slightly adjusted this but it's nothing major and the outputs/results are mostly the same.    

NOTE: from the paper:
To generate the force-directed layouts of the haemato-endothelial landscapes, 
we recomputed HVGs for each subset as well as batch correction of PCA manifolds (we again retained the top 50 principal components). 
We then built a KNN graph (K=50) and used ForceAtlas2 (Jacomy et al., 2014) implementation of force-directed layouts included in scanpy.
Its important to note that they batch correct. I'm thinking that if i run BBKNN it might resolve some of the weird plotting issues i've been having. 
"""

# Run Preprocessing and work from there.
# Log Normalisation
# Set soupX as X and then normalise that
reference_trajectory.X = reference_trajectory.layers["raw_counts"] # You should've done this earlier but this is a check
# control
scales_counts = sc.pp.normalize_total(reference_trajectory, target_sum=1e4, inplace=False) # 10000 counts 
# log1p transform
reference_trajectory.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# Highly variable gene selection: Had been done in the original dataset but for this, i'm going to ignore that and rerun.
# Here, we dont yet subset data to avoid loss of genes. I'll subset in a bit.
reference_trajectory.X = reference_trajectory.layers["log1p_norm"]

sc.pp.highly_variable_genes(reference_trajectory, n_top_genes=2000, inplace=True) # might remove the in-place bit
sc.pl.highly_variable_genes(reference_trajectory) # plot the HVG
# We could subset the data for the HVGs.

# Run PCA - set use_highly_variable to select only for HVG
# If we have some genes of interest, we can add them back in.
reference_HVG = reference_trajectory[:, reference_trajectory.var.highly_variable]
sc.pp.pca(reference_HVG, svd_solver="arpack") # Shouldnt need to run this again becuase we already have the X_pca?
# sc.pl.pca_variance_ratio(reference_HVG, log=True, n_pcs = 50)
reference_HVG

# The requirement is that the dimred model recapitulates the trajectory. My UMAP is fine but the actual one, based on the tutorial is not great.
# Need to fix but when reclustering, the cell types are all over the place. 
sc.pl.draw_graph(reference_HVG, color='celltype_extended_atlas')
# https://discourse.scverse.org/t/re-clustering-clusters-of-anndata/889/2
# Lets just check the object and examine  gene expression
markers = {"NMPs":["Sox2","T"],
           "NMPs/Mesoderm-biased":["Epha5"],
            "Somitic mesoderm":["Tbx6"],
            "Spinal cord progenitors":["Pax6"]
            }

for cell_type, genes in markers.items():
    print(f"marker gene of {cell_type}")
    sc.pl.draw_graph(reference_HVG, color=genes, use_raw=False, ncols=2)
    plt.show()


# Pseudotime calculation for CellOracle. 
# Decided to avoid renormalisation and just run with the subsetted object. Changes can be made in future.
import copy
import glob
import time
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm.auto import tqdm

#import time
import celloracle as co
from celloracle.applications import Pseudotime_calculator
co.__version__

# Instantiate pseudotime object using anndata object.
pt = Pseudotime_calculator(adata=reference_HVG,
                           obsm_key="X_draw_graph_fa", # Dimensional reduction data name
                           cluster_column_name="celltype_extended_atlas" # Clustering data name
                           )

# Check data
pt.plot_cluster(fontsize=10) # Tricky as the dots are super big.

# Define lineages 
mesoderm_lineage = ['Caudal mesoderm','NMPs','NMPs/Mesoderm-biased','Somitic mesoderm','Presomitic mesoderm',
                    'Posterior somitic tissues','Sclerotome','Lateral plate mesoderm'] 
neural_lineage = ['NMPs','Spinal cord progenitors','Dorsal spinal cord progenitors']

# Make a dictionary
lineage_dictionary = {"Lineage_meso": mesoderm_lineage,
           "Lineage_neural": neural_lineage}

# Input lineage information into pseudotime object
pt.set_lineage(lineage_dictionary=lineage_dictionary)

# Visualize lineage information
pt.plot_lineages()

# Define the root cell: i.e What is the starting cell.
# Hope is that we can select a "middle NMP" which is neither neural or mesodermal. This should have roughly equal T/Sox2.
# How fair is this? The NMC domain is extremely complex, heterogeneity is an important consideration
# Show interactive plot using plotly. Please make sure that plotly is installed.
try:
    import plotly.express as px
    def plot(adata, embedding_key, cluster_column_name):
        embedding = adata.obsm[embedding_key]
        df = pd.DataFrame(embedding, columns=["x", "y"])
        df["cluster"] = adata.obs[cluster_column_name].values
        df["label"] = adata.obs.index.values
        fig = px.scatter(df, x="x", y="y", hover_name=df["label"], color="cluster")
        fig.show()

    plot(adata=pt.adata,
         embedding_key=pt.obsm_key,
         cluster_column_name=pt.cluster_column_name)
except:
    print("Plotly not found in your environment. Did you install plotly? Please read the instruction above.")


# 388740 looks decent
