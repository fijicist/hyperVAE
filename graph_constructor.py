"""
Graph Constructor for HyperVAE Jet Dataset Preprocessing

This module constructs PyTorch Geometric (PyG) graph datasets from jet physics data
(JetNet or EnergyFlow) for training the HyperVAE model. It handles particle feature
engineering, normalization, EEC (Energy-Energy Correlator) computation, and graph
construction with optional hyperedge support.

Key Features:
-------------
1. **Data Loading**: Supports JetNet and EnergyFlow datasets
2. **Feature Engineering**: Converts particle kinematics (pt, eta, phi) to 4-momentum (E, px, py, pz)
3. **Normalization**: Configurable z-score or min-max normalization with statistics storage
4. **EEC Computation**: Calculates 2-point and n-point Energy-Energy Correlators
5. **Graph Construction**: Builds fully-connected particle graphs with edge features
6. **Hyperedges**: Optional n-point hyperedges for higher-order correlations
7. **Memory Optimization**: Batch saving for large datasets (every 85k graphs)

Output Format:
--------------
PyG Data objects with:
- `x`: Node features (normalized E, px, py, pz)
- `edge_index`: Fully-connected adjacency (all particle pairs)
- `edge_attr`: Edge features (2pt_EEC, ln_delta, ln_k_T, ln_z, ln_m2)
- `hyperedge_index`: N-point particle combinations (if enabled)
- `hyperedge_attr`: N-point EEC features (if enabled)
- `y`: Jet-level labels (type, pt, eta, mass)
- `particle_norm_stats`: Normalization statistics for particles
- `jet_norm_stats`: Normalization statistics for jet features
- `edge_norm_stats`: Normalization statistics for edge features
- `hyperedge_norm_stats`: Normalization statistics for hyperedge features

Configuration:
--------------
Edit GRAPH_CONSTRUCTION_CONFIG dictionary:
- `output_dir`: Where to save generated .pt files
- `N`: Number of jets to process
- `dataset`: 'jetnet' or 'energyflow'
- `normalization_method`: 'zscore' or 'minmax'
- `eec_prop`: [[N-point orders], bins, (R_Lmin, R_Lmax)]
- `additional_edge_attrs`: 'eec_without_charges' for 2-point EEC
- `additional_hypergraph_attrs`: 'n_point_hyperedges' for n-point EEC

Usage:
------
    python graph_constructor.py

Or import and call:
    from graph_constructor import _construct_particle_graphs_pyg, GRAPH_CONSTRUCTION_CONFIG
    
    _construct_particle_graphs_pyg(
        output_dir='./data/real/',
        N=18000,
        dataset='jetnet',
        normalization_method='zscore',
        ...
    )

Normalization Statistics:
--------------------------
All normalization statistics are stored in each PyG Data object for reversibility:

- `particle_norm_stats`: {'E': stats, 'px': stats, 'py': stats, 'pz': stats}
- `jet_norm_stats`: {'pt': stats, 'eta': stats, 'mass': stats}
- `edge_norm_stats`: {'2pt_EEC': stats, 'ln_delta': stats, 'ln_k_T': stats, 'ln_z': stats, 'ln_m2': stats}
- `hyperedge_norm_stats`: {'3pt_EEC': stats, '4pt_EEC': stats, ...}

Each stats dict contains: {'mean': float, 'std': float, 'method': 'zscore'} or
                         {'min': float, 'max': float, 'method': 'minmax'}

Physics Background:
-------------------
- **EEC (Energy-Energy Correlator)**: Observable measuring energy flow correlations
  between particles at angular distance ΔR. Used for jet substructure studies.
- **N-point EEC**: Generalization to N particles, captures higher-order correlations.
- **Jet clustering**: Optional reclustering with FastJet (anti-kt, R=0.4).
- **Edge features**: ln(ΔR), ln(k_T), ln(z), ln(m²) capture IRC-safe observables.

Memory Considerations:
----------------------
- Default config targets 4GB VRAM (GTX 1650Ti)
- Graphs saved in batches of 85,000 to avoid memory overflow
- Temporary variables deleted and gc.collect() called after normalization
- Edge features collected then normalized post-hoc to save memory

See Also:
---------
- `utils.py`: Helper functions (normalize_array, construct_n_point_hyperedges, etc.)
- `data/bipartite_dataset.py`: Dataset loader that reads these .pt files
- `config.yaml`: Model training configuration (separate from graph construction)
"""

import os
import gc
import tqdm
import numpy as np
import energyflow
import torch
import torch_geometric

import fastjet
import awkward as ak
import matplotlib.pyplot as plt

from utils import get_eec_ls_values, plot_jet_kinematics, reclusterJets, OneHotEncodeType, normalize_array, construct_n_point_hyperedges, plot_eec_values

from jetnet.datasets import JetNet

import time 
import networkx as nx
from torch_geometric.utils import to_networkx


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - User-Editable Parameters
# ═══════════════════════════════════════════════════════════════════════════════

GRAPH_CONSTRUCTION_CONFIG = {
    # Output directory for generated graphs
    'output_dir': './data/real/',
    
    # Graph structure (typically ['fully_connected'])
    'graph_structures': ['fully_connected'],
    
    # Number of jets to process
    'N': 10,
    
    # Dataset source: 'jetnet' or 'energyflow'
    'dataset': 'jetnet',
    
    # Whether to recluster jets (typically False for JetNet)
    'recluster_jets': False,
    
    # Normalization method for features
    # Options: 'zscore' (z-score normalization) or 'minmax' (min-max to [0,1])
    'normalization_method': 'zscore',
    
    # EEC (Energy-Energy Correlator) properties
    # Format: [[N-point orders], bins, (R_Lmin, R_Lmax)]
    # N-point orders: [2, 3, 4, ...] for 2-point, 3-point, 4-point EECs
    # bins: number of bins for EEC histograms
    # (R_Lmin, R_Lmax): angular distance range
    'eec_prop': [[2, 3], 200, (1e-4, 2)],
    
    # Additional feature flags
    'additional_node_attrs': None,
    'additional_edge_attrs': 'eec_without_charges',  # Options: None, 'eec_with_charges', 'eec_without_charges'
    'additional_graph_attrs': None,
    'additional_hypergraph_attrs': 'n_point_hyperedges',  # Options: None, 'n_point_hyperedges'
    
    # JetNet dataset arguments
    'data_args_jetnet': {
        "jet_type": ["q", "g", "t"],  # Quark, gluon, top jets
        "data_dir": "datasets/jetnet",
        "particle_features": ["ptrel", "etarel", "phirel", "mask"],
        "num_particles": 30,  # Maximum particles per jet
        "jet_features": ["type", "pt", "eta", "mass"],
        "download": True,
    }
}

# ═══════════════════════════════════════════════════════════════════════════════


def _construct_particle_graphs_pyg(
        output_dir,
        graph_structures,
        N=500000,
        dataset='jetnet',
        recluster_jets=False,
        normalization_method='zscore',  # 'zscore' or 'minmax'
        eec_prop=[[2, 3, 4, 5, 6, 7, 8, 9], 500, (1e-3, 2)], # [N, bins, (R_Lmin, R_Lmax)]
        additional_node_attrs=None,
        additional_edge_attrs=None, # None or eec_with_pids or eec_with_charges or eec_without_charges
        additional_graph_attrs=None,
        additional_hypergraph_attrs=None,
        data_args_jetnet = {
            "jet_type": ["q", "g", "t"],  # gluon and light quark jets
            "data_dir": "datasets/jetnet",
            # these are the default particle features, written here to be explicit
            "particle_features": ["ptrel", "etarel", "phirel", "mask"],
            "num_particles": 30,  # we retain only the 30 highest pT particles for this demo
            "jet_features": ["type", "pt", "eta", "mass"],
            # "particle_normalisation": FeaturewiseLinear(
            #     normal=True, normalise_features=[False, False, False]
            # ),
            # pass our function as a transform to be applied to the jet features
            # "jet_transform": OneHotEncodeType,
            "download": True,
        }
):
    '''
    Construct a list of PyG graphs for the particle-based GNNs, loading from the energyflow dataset

    Graph structure:
        - Nodes: particle four-vectors
        - Edges: no edge features
        - Connectivity: fully connected 
    '''
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENERGYFLOW DATASET LOADING (legacy, has known bugs)
    # ═══════════════════════════════════════════════════════════════════════════
    if dataset == 'energyflow':
        print(f'Constructing PyG particle graphs from energyflow dataset...')

        # Load dataset
        X, y = energyflow.qg_jets.load(N, pad=False)
        # energyflow.utils.remap_pids(X)

        # Store original features before transformation
        old_X = [0 for _ in range(len(X))]

        
        if reclusterJets:
        
            # Reclustering the jets using fastjet to check the clustering
            print(f'  Reclustering jets using fastjet...')
            
            inclusive_jets = [[0, 0, 0, 0] for _ in range(len(X))]

            for i in range(len(X)):

                # Remove zero-padded rows
                X[i] = X[i][~np.all(X[i] == 0, axis=1)]

                # Remove NaN values
                X[i] = X[i][~np.isnan(X[i]).any(axis=1)]

                # Track particle and jet energies for feature engineering
                energy_values = [0, 0]

                X[i] = energyflow.p4s_from_ptyphipids(X[i])
                X[i] = X[i].astype(np.float64)
            
                # Input to fastjet as an awkward array
                particleAwk = ak.zip({"px": X[i][:, 1], "py": X[i][:, 2], "pz": X[i][:, 3], "E": X[i][:, 0]})
            
                # Reclustering the jets
                reclustered_jets, inclusive_jet = reclusterJets(particleAwk, R=0.4, pt_cut=0)

                # For jet kinematics plot
                inclusive_jets[i][1] = ak.to_numpy(ak.unzip(inclusive_jet))[0]
                inclusive_jets[i][2] = ak.to_numpy(ak.unzip(inclusive_jet))[1]
                inclusive_jets[i][3] = ak.to_numpy(ak.unzip(inclusive_jet))[2]
                inclusive_jets[i][0] = ak.to_numpy(ak.unzip(inclusive_jet))[3]

                # Storing the jet kinematics in a list
                inclusive_jets[i] = [arr[0] for arr in inclusive_jets[i]]

                # Storing the jet E
                energy_values[1] = inclusive_jets[i][0]

                # For particle graphs
                X[i][:, 1] = ak.to_numpy(ak.unzip(reclustered_jets))[0]
                X[i][:, 2] = ak.to_numpy(ak.unzip(reclustered_jets))[1]
                X[i][:, 3] = ak.to_numpy(ak.unzip(reclustered_jets))[2]
                X[i][:, 0] = ak.to_numpy(ak.unzip(reclustered_jets))[3]

                # Storing the particle E
                energy_values[0] = X[i][:, 0]

                # Retrieving the particle coordinates in hadronic coordinates (pt, y, phi)
                X[i] = energyflow.ptyphims_from_p4s(X[i], mass=False)

                #Retrieving the Jet coordinates in hadronic coordinates (pt, y, phi)
                inclusive_jets[i] = energyflow.ptyphims_from_p4s(inclusive_jets[i], mass=False)


                # Making the new features according to the jet tagging papers
                old_X[i] = X[i] # Storing the old features of X (particles)
                old_X[i] = np.array(old_X[i])

                X[i] = np.zeros((len(X[i]), 7)) # Making a new array to store the new features

                # Storing the old features in the new array
                X[i][:, 0] = old_X[i][:, 1] - inclusive_jets[i][1] # delta_y
                X[i][:, 1] = old_X[i][:, 2] - inclusive_jets[i][2] # delta_phi
                X[i][:, 2] = np.log(old_X[i][:, 0]) # log(pt)
                X[i][:, 3] = np.log(energy_values[0]) # particle E
                X[i][:, 4] = np.log(old_X[i][:, 0] / inclusive_jets[i][0]) # log(pt / jet pt)
                X[i][:, 5] = np.log(energy_values[0] / energy_values[1]) # log(E / jet E)
                X[i][:, 6] = np.sqrt(X[i][:, 0]**2 + X[i][:, 1]**2) # delta_R

                # deleting the mass column from the jets
                # X[i] = np.delete(X[i], 3, 1)

            # plot_jet_kinematics(inclusive_jets)
            
            print("  Reclustering done.")


        # Preprocess by normalizing features
        for i, x in enumerate(tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X))):
            mask = ~np.isclose(x[:, 0], 0)
            
            # Apply the mask to eliminate rows with 0 values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with 0 values in old_X[i]
            old_X[i] = old_X[i][mask]

            # Create a mask to identify rows without NaN values in X[i]
            mask = ~np.isnan(X[i]).any(axis=1)

            # Apply the mask to eliminate rows with NaN values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with NaN values in old_X[i]
            old_X[i] = old_X[i][mask]

            # Per-jet z-score normalization
            X[i] = (X[i] - np.average(X[i], axis=0)) / np.std(X[i], axis=0)


    # ═══════════════════════════════════════════════════════════════════════════
    # JETNET DATASET LOADING (recommended)
    # ═══════════════════════════════════════════════════════════════════════════
    if dataset == 'jetnet':
        print(f'Constructing PyG particle graphs from JetNet dataset...')

        # Load dataset (X: particle features, y: jet-level labels)
        X, y = JetNet(**data_args_jetnet)[:N]
        X = X.numpy()
        y = y.numpy()#[:, 0].astype(int)

        # Filter out zero-padded particles
        result = []

        for i in range(X.shape[0]):
            # Filter out zero-padded rows
            non_zero_particles = X[i][~np.all(X[i] == 0, axis=1)]
            result.append(non_zero_particles)

        X = result

        # Making an empty list to store the old features
        old_X = [0 for _ in range(len(X))]

        # Store original features (pt, eta, phi) before transforming to 4-momentum
        old_X = [0 for _ in range(len(X))]

        for i in range(len(X)):

            # Making the new features according to the jet tagging papers

            # Storing the old features of X (particles)
            old_X[i] = np.array(X[i][:, :3])
            mask = X[i][:, 3].astype(bool)
            old_X[i] = old_X[i][mask]

            # Convert relative pt to absolute pt using jet pt
            old_X[i][:, 0] = old_X[i][:, 0] * y[i][1]

            # X[i] = np.zeros((len(X[i][mask]), 7)) # Making a new array to store the new features
            X[i] = np.zeros((len(X[i][mask]), 4)) # Making a new array to store the new features

            # Storing the old features in the new array
            # X[i][:, 0] = np.log(old_X[i][:, 0]) # log(pt)
            # X[i][:, 1] = old_X[i][:, 1] # rel_eta
            # X[i][:, 2] = old_X[i][:, 2] # rel_phi
            # X[i][:, 3] = np.log(old_X[i][:, 0] * np.cosh(old_X[i][:, 1] + y[i][2])) # log(particle E)

            # X[i][:, 4] = np.log(old_X[i][:, 0] / y[i][1]) # log(pt / jet pt)
            # X[i][:, 5] = X[i][:, 3] - np.log(np.sqrt(y[i][1]**2 + y[i][3]**2)) # log(E / jet E)
            # X[i][:, 6] = np.sqrt(X[i][:, 0]**2 + X[i][:, 1]**2) # delta_R

            # New features: E, px, py , pz
            X[i][:, 0] = old_X[i][:, 0] * np.cosh(old_X[i][:, 1] + y[i][2]) # particle E
            X[i][:, 1] = old_X[i][:, 0] * np.cos(old_X[i][:, 2]) # particle px
            X[i][:, 2] = old_X[i][:, 0] * np.sin(old_X[i][:, 2]) # particle py
            X[i][:, 3] = old_X[i][:, 0] * np.sinh(old_X[i][:, 1] + y[i][2]) # particle pz

            X[i] = np.array(X[i])
            old_X[i] = np.array(old_X[i])


        # Clean up: remove zeros and NaNs
        for i, x in enumerate(tqdm.tqdm(X, desc='  Preprocessing jets', total=len(X))):
            mask = ~np.isclose(x[:, 0], 0)
            
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with 0 values in old_X[i]
            old_X[i] = old_X[i][mask]

            # Create a mask to identify rows without NaN values in X[i]
            mask = ~np.isnan(X[i]).any(axis=1)

            # Apply the mask to eliminate rows with NaN values in X[i]
            X[i] = X[i][mask]

            # Use the same mask to eliminate rows with NaN values in old_X[i]
            old_X[i] = old_X[i][mask]

            # Making type int labels
            y[i][0] = int(y[i][0])

            # Taking the log of jet pt and jet mass
            y[i][1] = np.log(y[i][1])  # log(jet pt)
            y[i][3] = np.log(y[i][3])  # log(jet mass)

        # ───────────────────────────────────────────────────────────────────────
        # Global Normalization of Particle Features (E, px, py, pz)
        # ───────────────────────────────────────────────────────────────────────
        # Flatten all particle features across all jets for global statistics
        E_features = np.concatenate([x[:, 0] for x in X])
        px_features = np.concatenate([x[:, 1] for x in X])
        py_features = np.concatenate([x[:, 2] for x in X])
        pz_features = np.concatenate([x[:, 3] for x in X])

        jet_pt_features = np.array(list(y[i][1] for i in range(len(y))))
        jet_eta_features = np.array(list(y[i][2] for i in range(len(y))))
        jet_mass_features = np.array(list(y[i][3] for i in range(len(y))))

        # Normalize and store statistics (for reversibility during generation)
        E_norm, E_stats = normalize_array(E_features, method=normalization_method, return_stats=True)
        px_norm, px_stats = normalize_array(px_features, method=normalization_method, return_stats=True)
        py_norm, py_stats = normalize_array(py_features, method=normalization_method, return_stats=True)
        pz_norm, pz_stats = normalize_array(pz_features, method=normalization_method, return_stats=True)
        
        jet_pt_norm, jet_pt_stats = normalize_array(jet_pt_features, method=normalization_method, return_stats=True)
        jet_eta_norm, jet_eta_stats = normalize_array(jet_eta_features, method=normalization_method, return_stats=True)
        jet_mass_norm, jet_mass_stats = normalize_array(jet_mass_features, method=normalization_method, return_stats=True)
        
        # Store in dictionaries with descriptive keys
        particle_norm_stats = {
            'E': E_stats,
            'px': px_stats,
            'py': py_stats,
            'pz': pz_stats
        }
        
        jet_norm_stats = {
            'pt': jet_pt_stats,
            'eta': jet_eta_stats,
            'mass': jet_mass_stats
        }
        
        # Reconstruct normalized values from flattened arrays back to per-jet format
        idx = 0
        for i in range(len(X)):
            n_particles = X[i].shape[0]
            X[i][:, 0] = E_norm[idx:idx + n_particles]
            X[i][:, 1] = px_norm[idx:idx + n_particles]
            X[i][:, 2] = py_norm[idx:idx + n_particles]
            X[i][:, 3] = pz_norm[idx:idx + n_particles]
            idx += n_particles
        
        # Apply normalized jet-level features
        for i in range(len(X)):
            y[i][1] = jet_pt_norm[i]
            y[i][2] = jet_eta_norm[i]
            y[i][3] = jet_mass_norm[i]

        # Delete temporary variables to free memory
        del E_features, px_features, py_features, pz_features
        del E_norm, px_norm, py_norm, pz_norm
        del jet_pt_features, jet_eta_features, jet_mass_features
        del jet_pt_norm, jet_eta_norm, jet_mass_norm
        gc.collect()

        print("  Normalization of node features done.")


        # Optional: one-hot encode jet types (currently disabled)
        # y[:, 0] = OneHotEncodeType(y)[:, 0]
        # y[:, 0] = y[:, 0].astype(int)


        # plot_jet_kinematics(X, input_type='hadronic')
        
        # plotting
        # fig, axs = plt.subplots(1, 7, figsize=(20, 5))
        # 
        # jet_list = [np.array([]) for _ in range(7)]
        # for i in range(7):
        #     for j in range(len(X)):
        #         jet_list[i] = np.append(jet_list[i], X[j][:, i])
        #
        # for i in range(7):
        #     axs[i].hist(jet_list[i], bins=100)
        #     axs[i].set_title(f'Histogram of X[:, :, {i}]')
        #
        # plt.tight_layout()
        # plt.savefig('scatter_plot.png')
        # exit()

    # Calculate EnergyEnergyCorrelation (EEC) features and normalization statistics
    edge_norm_stats = {}  # Dictionary with keys: '2pt_EEC', 'ln_delta', 'ln_k_T', 'ln_z', 'ln_m2'
    eec2_values = None
    
    if additional_edge_attrs == 'eec_with_charges':
        print("  Calculating EEC features with charges...")
        additional_edge_attrs = []
        if eec_prop[0][0] == 2:
            # Compute 2-point EEC histogram across all jets
            additional_edge_attrs.append(get_eec_ls_values(old_X, N=eec_prop[0][0], bins=eec_prop[1], axis_range=eec_prop[2]))
            
            # Compute normalization statistics immediately (needed for per-edge normalization later)
            print("  Computing normalization statistics for 2-point EEC...")
            eec2_values = np.array(list(additional_edge_attrs[0].get_hist_errs(0, False)[0]))
            with np.errstate(divide='ignore'):
                log_eec2 = np.where(eec2_values > 0, np.log(eec2_values), 0.0)
            _, stats_2pt = normalize_array(log_eec2, method=normalization_method, return_stats=True)
            edge_norm_stats['2pt_EEC'] = stats_2pt

    if additional_edge_attrs == 'eec_without_charges':
        print("  Calculating EEC features without charges...")
        additional_edge_attrs = []
        if eec_prop[0][0] == 2:
            additional_edge_attrs.append(get_eec_ls_values(old_X, N=eec_prop[0][0], bins=eec_prop[1], axis_range=eec_prop[2]))
            
            # Compute 2-point EEC normalization statistics immediately
            print("  Computing normalization statistics for 2-point EEC...")
            eec2_values = np.array(list(additional_edge_attrs[0].get_hist_errs(0, False)[0]))
            with np.errstate(divide='ignore'):
                log_eec2 = np.where(eec2_values > 0, np.log(eec2_values), 0.0)
            _, stats_2pt = normalize_array(log_eec2, method=normalization_method, return_stats=True)
            edge_norm_stats['2pt_EEC'] = stats_2pt

    # Calculate n-point EEC features and normalization statistics
    hyperedge_norm_stats = {}  # Will store: '3pt_EEC', '4pt_EEC', etc.
    
    if additional_hypergraph_attrs == 'n_point_hyperedges':
        print("  Calculating EEC features for hyperedges...")
        additional_hypergraph_attrs = []
        
        for n_point in eec_prop[0]:
            if n_point == 2:
                continue  # Already computed above
            else:
                # Compute n-point EEC histogram
                eec_n_hist = get_eec_ls_values(old_X, N=n_point, bins=eec_prop[1], axis_range=eec_prop[2])
                additional_hypergraph_attrs.append(eec_n_hist)
                
                # Compute normalization statistics
                print(f"  Computing normalization statistics for {n_point}-point EEC...")
                eec_n_values = np.array(list(eec_n_hist.get_hist_errs(0, False)[0]))
                
                # Divide by 2-point EEC to reduce correlation: log(EEC_n / EEC_2)
                if eec2_values is not None:
                    eec_n_values = np.divide(eec_n_values, eec2_values,
                                            out=np.zeros_like(eec_n_values),
                                            where=eec2_values != 0)
                
                # Log transform
                with np.errstate(divide='ignore'):
                    eec_n_values = np.where(eec_n_values > 0, np.log(eec_n_values), 0.0)
                
                # Normalize and store with descriptive key ('3pt_EEC', '4pt_EEC', etc.)
                _, stats_npt = normalize_array(eec_n_values, method=normalization_method, return_stats=True)
                hyperedge_norm_stats[f'{n_point}pt_EEC'] = stats_npt

    # Optional: plot EEC distributions for verification
    if additional_edge_attrs or additional_hypergraph_attrs:
        print("  Plotting EEC values...")
        plot_eec_values(additional_edge_attrs, additional_hypergraph_attrs, eec_prop, output_dir)

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH CONSTRUCTION LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    # RAM Memory optimization strategy (hyperedge computation is CPU-bound):
    # 1. Flush RAM every 1000 graphs during construction (gc.collect())
    # 2. Flush RAM after saving batches (every 85k graphs)  
    # 3. Flush RAM after constructing hyperedges for each graph
    # 4. This prevents OOM when processing large datasets with hyperedges
    # ═══════════════════════════════════════════════════════════════════════════
    for graph_structure in graph_structures:
        graph_key = f'particle__{graph_structure}'
        all_edge_features = []
        saved_filenames = []
        graph_list = []

        args = [(x, y[i], old_X[i], particle_norm_stats, jet_norm_stats, edge_norm_stats, hyperedge_norm_stats) for i, x in enumerate(X)] # Using old_X to store the old features
        for i, arg in enumerate(tqdm.tqdm(args, desc=f'  Constructing PyG graphs: {graph_key}', total=len(args))):

            graph = _construct_particle_graph_pyg(arg, additional_node_attrs, additional_edge_attrs, additional_graph_attrs, additional_hypergraph_attrs, normalization_method)
            graph_list.append(graph)
            
            # Collect edge features for normalization (index 1 onwards only)
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                all_edge_features.append(graph.edge_attr[:, 1:].cpu().numpy())
            
            # Flush RAM periodically (every 1000 graphs) to prevent gradual OOM
            # Force garbage collection to free up memory from hyperedge computations
            if (i + 1) % 1000 == 0:
                gc.collect()

            # Save to file every 85000 iterations
            if (i + 1) % 85000 == 0:
                partial_graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}_part_{i // 85000 + 1}.pt")
                torch.save(graph_list, partial_graph_filename)
                print(f'  Saved PyG graphs to {partial_graph_filename}.')
                saved_filenames.append(partial_graph_filename)
                graph_list = []
                
                # Flush RAM after batch save to prevent OOM
                # Free up memory from processed graphs
                gc.collect()
                print(f'  RAM flushed after saving batch {i // 85000 + 1}.')

        # Save any remaining graphs
        if graph_list:
            final_graph_filename = os.path.join(output_dir, f"graphs_pyg_{graph_key}_final.pt")
            torch.save(graph_list, final_graph_filename)
            print(f'  Saved PyG graphs to {final_graph_filename}.')
            saved_filenames.append(final_graph_filename)
            
        # Compute normalization statistics for additional edge features (ln_delta, ln_k_T, ln_z, ln_m2)
        # These are at indices 0-3 in stacked_edges (index 0 is 2pt_EEC which is already normalized, so we skip it)
        if all_edge_features:
            print(f'  Computing normalization statistics for additional edge features (ln_delta, ln_k_T, ln_z, ln_m2)...')
            stacked_edges = np.vstack(all_edge_features)
            
            print(f'    Collected edge features shape: {stacked_edges.shape}')
            print(f'    Number of graphs processed: {len(all_edge_features)}')
            print(f'    Normalization method: {normalization_method}')
            
            # Feature names for indices 0, 1, 2, 3 in stacked_edges (which correspond to indices 1, 2, 3, 4 in edge_attr)
            additional_feature_names = ['ln_delta', 'ln_k_T', 'ln_z', 'ln_m2']
            
            # Compute and store statistics for each additional edge feature
            print(f'    Computing normalization statistics for each feature...')
            for idx, feature_name in enumerate(additional_feature_names):
                feature_values = stacked_edges[:, idx]
                normalized_values, stats = normalize_array(feature_values, method=normalization_method, return_stats=True)
                edge_norm_stats[feature_name] = stats
                print(f'      {feature_name}: stats = {stats}')
                
                # Store normalized values back into stacked_edges for application to graphs
                stacked_edges[:, idx] = normalized_values
            
            print(f'    edge_norm_stats keys: {list(edge_norm_stats.keys())}')
            
            # Convert to torch tensor for graph updates
            stacked_edges_tensor = torch.tensor(stacked_edges, dtype=torch.float32)
            
            # Normalize edge features in all saved files (indices 1 onwards only)
            print(f'    Applying normalization to {len(saved_filenames)} saved file(s)...')
            edge_idx = 0  # Track position in stacked_edges
            for filename in tqdm.tqdm(saved_filenames, desc=f'  Normalizing edge features: {graph_key}'):
                graphs = torch.load(filename)
                for graph in graphs:
                    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                        n_edges = graph.edge_attr.shape[0]
                        # Replace unnormalized edge features (indices 1:) with normalized values
                        graph.edge_attr[:, 1:] = stacked_edges_tensor[edge_idx:edge_idx + n_edges]
                        edge_idx += n_edges
                    
                    # Update edge_norm_stats in the graph to include additional features
                    if hasattr(graph, 'edge_norm_stats'):
                        graph.edge_norm_stats = edge_norm_stats
                torch.save(graphs, filename)
            
            print(f'  ✓ Edge features normalized for {graph_key}.')
            print(f'  ✓ Total edges processed: {edge_idx}')
            print(f'  ✓ Normalization statistics stored in edge_norm_stats with keys: {list(edge_norm_stats.keys())}')
            print(f'  ✓ All graphs updated with complete edge_norm_stats dictionary')
    print('Graph construction complete.')


def _construct_particle_graph_pyg(
        args,
        additional_node_attrs=None,
        additional_edge_attrs=None,
        additional_graph_attrs=None,
        additional_hypergraph_attrs=None,
        normalization_method='zscore'
):
    '''
    Construct a single PyG Data object for one jet.
    
    Creates a fully-connected graph with:
    - Nodes: particle 4-momenta (normalized E, px, py, pz)
    - Edges: 2-point EEC + additional features (ln_delta, ln_k_T, ln_z, ln_m2)
    - Hyperedges: n-point particle combinations with n-point EEC features (optional)
    - Graph label: jet type, pt, eta, mass
    
    All normalization statistics are attached for reversibility during generation.
    '''
    # Unpack arguments: x=normalized features, old_x=original (pt,eta,phi) for EEC, y=jet label
    x, y, old_x, particle_norm_stats, jet_norm_stats, edge_norm_stats, hyperedge_norm_stats = args


    # ───────────────────────────────────────────────────────────────────────────
    # Node Features and Graph Topology
    # ───────────────────────────────────────────────────────────────────────────
    x = x[~np.all(x == 0, axis=1)]  # Remove zero-padded particles
    node_features = torch.tensor(x, dtype=torch.float)

    # Fully-connected graph (all particle pairs)
    adj_matrix = np.ones((x.shape[0], x.shape[0])) - np.identity((x.shape[0]))
    row, col = np.where(adj_matrix)
    coo = np.array(list(zip(row, col)))
    edge_indices = torch.tensor(coo)
    edge_indices_long = edge_indices.t().to(torch.long).view(2, -1)

    # Graph-level label: [jet_type, log(pt), eta, log(mass)]
    graph_label = torch.tensor(y, dtype=torch.float32)


    # Add additional attributes if provided
    # if additional_node_attrs:
    #     graph.node_attrs = torch.tensor(additional_node_attrs, dtype=torch.float)

    if additional_edge_attrs:
        # Preallocate: [2pt_EEC, ln_delta, ln_k_T, ln_z, ln_m2]
        edge_features = [[] for _ in range(len(additional_edge_attrs) + 4)]

        # Normalize 2-point EEC histogram using precomputed statistics
        edge_attrs = []

        for i in range(len(additional_edge_attrs)):

            # Extract and log-transform EEC values
            eec_values = np.array(list(additional_edge_attrs[i].get_hist_errs(0, False)[0]))
            with np.errstate(divide='ignore'):
                eec_values = np.where(eec_values > 0, np.log(eec_values), 0.0)
            
            # Apply normalization using stored statistics
            if edge_norm_stats and '2pt_EEC' in edge_norm_stats:
                stats = edge_norm_stats['2pt_EEC']
                if stats['method'] == 'zscore':
                    normalized_eec = np.where(~np.isclose(eec_values, 0), 
                                             (eec_values - stats['mean']) / stats['std'], 0.0)
                elif stats['method'] == 'minmax':
                    if stats['max'] - stats['min'] != 0:
                        normalized_eec = (eec_values - stats['min']) / (stats['max'] - stats['min'])
                    else:
                        normalized_eec = np.zeros_like(eec_values)
                normalized_eec = np.nan_to_num(normalized_eec, nan=0.0)
            else:
                # Fallback: normalize without stats (shouldn't happen)
                normalized_eec = normalize_array(eec_values, method=normalization_method)
            
            edge_attrs.append(normalized_eec)

        # Loop over all particle pairs to compute per-edge features
        for i, j in zip(row, col):

            # Compute angular distance ΔR between particles i and j
            delta_y = old_x[i][1] - old_x[j][1]
            delta_phi_abs = abs(old_x[i][2] - old_x[j][2])
            delta_phi = delta_phi_abs if delta_phi_abs <= np.pi else 2 * np.pi - delta_phi_abs  # Handle wrap-around
            delta_R = np.sqrt(delta_y**2 + delta_phi**2)
            
            # if delta_R > 1.8:
            #     print(f"delta_R: {delta_R}, delta_y: {delta_y}, delta_phi: {delta_phi}")

            # Determine the bin for the edge value
            bin_index = np.digitize(delta_R, bins=additional_edge_attrs[0].bin_edges()) - 1

            # Calculate other log-transformed edge features
            # Δ (delta)
            delta = delta_R
            
            # k_T
            k_T = min(old_x[i][0], old_x[j][0]) * delta
            
            # z
            z = min(old_x[i][0], old_x[j][0]) / (old_x[i][0] + old_x[j][0])
            
            # m^2 (invariant mass squared)
            # calculate energy components
            E_a = old_x[i][0] * np.cosh(old_x[i][1])
            E_b = old_x[j][0] * np.cosh(old_x[j][1])
            
            # Calculate momentum components
            p_x_a = old_x[i][0] * np.cos(old_x[i][2])
            p_y_a = old_x[i][0] * np.sin(old_x[i][2])
            p_z_a = old_x[i][0] * np.sinh(old_x[i][1])
            
            p_x_b = old_x[j][0] * np.cos(old_x[j][2])
            p_y_b = old_x[j][0] * np.sin(old_x[j][2])
            p_z_b = old_x[j][0] * np.sinh(old_x[j][1])
            
            # Calculate invariant mass squared m^2
            m2 = (E_a + E_b)**2 - ((p_x_a + p_x_b)**2 + (p_y_a + p_y_b)**2 + (p_z_a + p_z_b)**2)
            
            # Log-transform features (IRC-safe)
            ln_delta = np.log(delta)
            ln_k_T = np.log(k_T)
            ln_z = np.log(z)
            ln_m2 = np.log(m2) if m2 > 0 else 0 # Avoid negative values

            # Get the histogram value for the bin
            for k in range(len(additional_edge_attrs) + 4):
                # EEC values
                if k < len(additional_edge_attrs):
                    edge_features[k].append(edge_attrs[k][bin_index])

                # Additional features
                elif k == len(additional_edge_attrs):
                    edge_features[k].append(ln_delta)
                elif k == len(additional_edge_attrs) + 1:
                    edge_features[k].append(ln_k_T)
                elif k == len(additional_edge_attrs) + 2:
                    edge_features[k].append(ln_z)
                elif k == len(additional_edge_attrs) + 3:
                    edge_features[k].append(ln_m2)


        # Convert edge features to tensor
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        edge_features_tensor = edge_features_tensor.t() # Transpose to dim (n_edges, n_features)


        # Construct graph as PyG data object with normalization statistics
        graph = torch_geometric.data.Data(
            x=node_features, 
            edge_index=edge_indices_long, 
            edge_attr=edge_features_tensor, 
            y=graph_label,
            particle_norm_stats=particle_norm_stats,  # Store particle normalization statistics
            jet_norm_stats=jet_norm_stats,  # Store jet-level normalization statistics
            edge_norm_stats=edge_norm_stats,  # Store edge (2-point EEC) normalization statistics
            hyperedge_norm_stats=hyperedge_norm_stats  # Store hyperedge normalization statistics (may be None)
        )

        # Returning all the tensors for normalizing the edge features later
        
        # # Visualizing the PyG graphs
        # nx_graph = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
        # nx.draw(nx_graph, with_labels=True, node_size=30, node_color='blue', edge_color='gray')
        # plt.show()
        # exit()

    # if additional_graph_attrs:
    #     graph.graph_attrs = torch.tensor(additional_graph_attrs, dtype=torch.float)

    if additional_hypergraph_attrs:
        # num_nodes is determined from the processed node features.
        num_nodes = x.shape[0]

        # Choose the desired order for the hyperedges (e.g., n=3 for 3-point, n=4 for 4-point, etc.)
        n_point = [i+3 for i in range(len(additional_hypergraph_attrs))]  # For HypergraphConv layer, has to be equal to no. of node features

        # Construct the N-point hyperedges with precomputed normalization
        start_time = time.perf_counter()
        hyperedge_index, hyperedge_attr = construct_n_point_hyperedges(
            num_nodes, old_x, additional_hypergraph_attrs, 
            n=n_point, 
            eec2=np.array(list(additional_edge_attrs[0].get_hist_errs(0, False)[0])),
            hyperedge_norm_stats=hyperedge_norm_stats,
            normalization_method=normalization_method
        )
        hyperedge_index = hyperedge_index.to_dense().type(torch.int)  # Convert to full binary coincidence matrix (dense tensor)
        end_time = time.perf_counter()
        print(f"Time taken to construct hyperedges: {end_time - start_time:.2f} seconds.")
        
        # Flush RAM after hyperedge construction to prevent OOM
        # Delete temporary variables and force garbage collection
        del start_time, end_time
        gc.collect()
        
        # print(hyperedge_index, hyperedge_attr, hyperedge_index.shape, hyperedge_attr.shape)
        # print(node_features.shape, edge_indices_long.shape, edge_features_tensor.shape, graph_label.shape)

    # Construct graph as PyG data object
    if additional_edge_attrs and additional_hypergraph_attrs:
        graph = torch_geometric.data.Data(
            x=node_features,          # Node features.
            edge_index=edge_indices_long,    # Normal pairwise connectivity.
            edge_attr=edge_features_tensor,  # Normal edge features.
            hyperedge_index=hyperedge_index,   # Hypergraph incidence (for N-point hyperedges).
            hyperedge_attr=hyperedge_attr,     # N-point EEC features for each hyperedge.
            y=graph_label,             # Graph label.
            particle_norm_stats=particle_norm_stats,  # Store particle normalization statistics
            jet_norm_stats=jet_norm_stats,  # Store jet-level normalization statistics
            edge_norm_stats=edge_norm_stats,  # Store edge (2-point EEC) normalization statistics
            hyperedge_norm_stats=hyperedge_norm_stats  # Store hyperedge (n-point EEC) normalization statistics
        )


    else:
        graph = torch_geometric.data.Data(
            x=node_features, 
            edge_index=edge_indices_long, 
            edge_attr=None, 
            y=graph_label,
            particle_norm_stats=particle_norm_stats,  # Store particle normalization statistics
            jet_norm_stats=jet_norm_stats,  # Store jet-level normalization statistics
            edge_norm_stats=edge_norm_stats,  # Store edge normalization statistics
            hyperedge_norm_stats=hyperedge_norm_stats  # Store hyperedge normalization statistics
        )

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION - Construct graphs using configuration above
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _construct_particle_graphs_pyg(
        output_dir=GRAPH_CONSTRUCTION_CONFIG['output_dir'],
        graph_structures=GRAPH_CONSTRUCTION_CONFIG['graph_structures'],
        N=GRAPH_CONSTRUCTION_CONFIG['N'],
        dataset=GRAPH_CONSTRUCTION_CONFIG['dataset'],
        recluster_jets=GRAPH_CONSTRUCTION_CONFIG['recluster_jets'],
        normalization_method=GRAPH_CONSTRUCTION_CONFIG['normalization_method'],
        eec_prop=GRAPH_CONSTRUCTION_CONFIG['eec_prop'],
        additional_node_attrs=GRAPH_CONSTRUCTION_CONFIG['additional_node_attrs'],
        additional_edge_attrs=GRAPH_CONSTRUCTION_CONFIG['additional_edge_attrs'],
        additional_graph_attrs=GRAPH_CONSTRUCTION_CONFIG['additional_graph_attrs'],
        additional_hypergraph_attrs=GRAPH_CONSTRUCTION_CONFIG['additional_hypergraph_attrs'],
        data_args_jetnet=GRAPH_CONSTRUCTION_CONFIG['data_args_jetnet']
    )
