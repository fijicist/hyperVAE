"""
Utility Functions for HyperVAE Jet Data Preprocessing

This module provides helper functions for:
1. **EEC Calculation**: Energy-Energy Correlator computation for jet substructure
2. **Hyperedge Construction**: N-point particle combinations with angular distance features
3. **Normalization**: Configurable z-score or min-max normalization with statistics storage
4. **Jet Reclustering**: FastJet interface for anti-kt clustering
5. **Data Transforms**: One-hot encoding, feature engineering

Key Functions:
--------------
- `normalize_array()`: Normalize features with either z-score or min-max, return statistics
- `construct_n_point_hyperedges()`: Build N-point hyperedges with EEC features
- `get_eec_ls_values()`: Compute Energy-Energy Correlators using EEC library
- `reclusterJets()`: Recluster particles with FastJet
- `OneHotEncodeType()`: One-hot encode jet types (q/g/t)

See individual function docstrings for detailed usage.
"""

import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import eec
import fastjet
from sklearn.preprocessing import OneHotEncoder
import os

from joblib import Parallel, delayed, Memory
import numba as nb

from torch_geometric.data import Batch 

def custom_collate_fn(data_list):
    """Collate function to pad node features to maximum size in batch."""
    max_node_features = max([data.x.size(1) for data in data_list])
    
    for data in data_list:
        if data.x.size(1) < max_node_features:
            padding = max_node_features - data.x.size(1)
            data.x = torch.cat([data.x, torch.zeros(data.x.size(0), padding)], dim=1)
    return Batch.from_data_list(data_list)

# Shared memory for large arrays (speeds up parallel EEC computation)
memory = Memory(location='/tmp/joblib_cache', verbose=0)

# Precompute RL_n for a hyperedge using Numba JIT
@nb.njit(fastmath=True)
def compute_RL_n(coords):
    """
    Compute RL_n (maximum pairwise angular distance) for a given hyperedge.
    
    Args:
        coords (np.ndarray): Array of shape (n, 2), where each row contains [delta_y, delta_phi].
    
    Returns:
        float: Maximum pairwise angular distance RL_n.
    """
    n = coords.shape[0]
    max_distance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy = coords[i, 0] - coords[j, 0]
            dphi = abs(coords[i, 1] - coords[j, 1])
            if dphi > np.pi:
                dphi = 2 * np.pi - dphi
            distance = np.sqrt(dy**2 + dphi**2)
            if distance > max_distance:
                max_distance = distance
    return max_distance

# Process a single hyperedge using Numba-compatible operations
def process_hyperedge(hyperedge, old_x, precomputed_bin_edges, precomputed_hist_values):
    """
    Process a single hyperedge to compute RL_n and EEC values.

    Args:
        hyperedge (tuple): Indices of nodes in the hyperedge.
        old_x (np.ndarray): Node feature matrix of shape [num_nodes, d].
        precomputed_bin_edges (list): Precomputed bin edges for all histograms.
        precomputed_hist_values (list): Precomputed histogram values for all histograms.

    Returns:
        tuple: Hyperedge and its computed EEC values.
    """
    # Extract coordinates for the nodes in this hyperedge
    coords = old_x[list(hyperedge), 1:3]  # Extract [delta_eta, delta_phi]
    
    # Compute RL_n using the optimized Numba function
    RL_n = compute_RL_n(coords)
    
    # Compute EEC values for this hyperedge
    eec_vals = []
    for bin_edges, hist_values in zip(precomputed_bin_edges, precomputed_hist_values):
        bin_index = np.digitize(RL_n, bins=bin_edges) - 1
        
        if 0 <= bin_index < len(hist_values):
            eec_vals.append(hist_values[bin_index])
        else:
            eec_vals.append(0.0)
    
    return hyperedge, eec_vals

def process_hyperedge_batch(hyperedge_batch, old_x_shared, precomputed_bin_edges,
                            precomputed_hist_values):
    """
    Process a batch of hyperedges.

    Args:
        hyperedge_batch (list): List of hyperedges to process.
        old_x_shared (np.ndarray): Shared memory-mapped node feature matrix.
        precomputed_bin_edges (list): Precomputed bin edges for all histograms.
        precomputed_hist_values (list): Precomputed histogram values for all histograms.

    Returns:
        list: List of processed hyperedges and their attributes.
    """
    results = []
    for hyperedge in hyperedge_batch:
        results.append(process_hyperedge(hyperedge, old_x_shared,
                                         precomputed_bin_edges,
                                         precomputed_hist_values))
    return results

def construct_n_point_hyperedges(num_nodes, old_x, additional_hypergraph_attrs, n, eec2, 
                                 hyperedge_norm_stats=None, normalization_method='zscore'):
    """
    Constructs n-point hyperedges with Energy-Energy Correlator features.
    
    Each hyperedge connects N particles and has a feature value from the N-point EEC histogram.
    Uses precomputed normalization statistics for consistency with 2-point EEC.

    Args:
        num_nodes (int): Number of particles in the jet
        old_x (np.ndarray): Particle features [num_nodes, 3] = [pt, eta, phi]
        additional_hypergraph_attrs (list): List of EEC histogram objects (one per N-point order)
        n (int or list): N-point order(s), e.g., [3, 4] for 3-point and 4-point
        eec2 (np.ndarray): 2-point EEC histogram values (for normalization)
        hyperedge_norm_stats (dict): Precomputed normalization stats with keys '3pt_EEC', '4pt_EEC', etc.
        normalization_method (str): 'zscore' or 'minmax' (fallback if stats not provided)

    Returns:
        hyperedge_index (torch.sparse): Binary incidence matrix [num_nodes, num_hyperedges]
        hyperedge_attr (torch.Tensor): Feature matrix [num_hyperedges, num_features]
    """
    # Ensure n is a list
    n_list = [n] if isinstance(n, int) else n

    # Cache large arrays in shared memory for parallel processing
    old_x_shared = memory.cache(np.array)(old_x)

    # Precompute static data from additional_hypergraph_attrs
    precomputed_bin_edges = [hist_obj.bin_edges() for hist_obj in additional_hypergraph_attrs]
    precomputed_hist_values = [np.array(list(hist_obj.get_hist_errs(0, False)[0]))
                                for hist_obj in additional_hypergraph_attrs]

    # Normalize precomputed histogram values: log(ENC/E2C)
    # If hyperedge_norm_stats are provided, use them; otherwise compute normalization
    for i in range(len(precomputed_hist_values)):
        # Divide by 2-point EEC
        precomputed_hist_values[i] = np.divide(precomputed_hist_values[i], eec2,
                                   out=np.zeros_like(precomputed_hist_values[i]),
                                   where=eec2 != 0)
        
        # Log transform
        with np.errstate(divide='ignore'):
            precomputed_hist_values[i] = np.where(precomputed_hist_values[i] > 0, 
                                                   np.log(precomputed_hist_values[i]), 0.0)
        
        # Apply normalization using precomputed stats if available
        # Determine n-point value for this index (skip 2, so starts at 3)
        n_point_vals = [val for val in n_list if val != 2]
        if i < len(n_point_vals):
            n_val = n_point_vals[i]
            stats_key = f'{n_val}pt_EEC'
            
            if hyperedge_norm_stats and stats_key in hyperedge_norm_stats:
                stats = hyperedge_norm_stats[stats_key]
                if stats['method'] == 'zscore':
                    precomputed_hist_values[i] = np.where(
                        ~np.isclose(precomputed_hist_values[i], 0),
                        (precomputed_hist_values[i] - stats['mean']) / stats['std'],
                        0.0
                    )
                elif stats['method'] == 'minmax':
                    if stats['max'] - stats['min'] != 0:
                        precomputed_hist_values[i] = (precomputed_hist_values[i] - stats['min']) / (stats['max'] - stats['min'])
                    else:
                        precomputed_hist_values[i] = np.zeros_like(precomputed_hist_values[i])
                precomputed_hist_values[i] = np.nan_to_num(precomputed_hist_values[i], nan=0.0)
            else:
                # Fallback: normalize without precomputed stats
                precomputed_hist_values[i] = normalize_array(precomputed_hist_values[i], method=normalization_method)

    # Cache precomputed data in shared memory
    precomputed_hist_values = memory.cache(list)(precomputed_hist_values)
    precomputed_bin_edges = memory.cache(list)(precomputed_bin_edges)

    all_hyperedges = []
    all_hyperedge_attrs = []

    # Loop over each n-value and iterate directly using the iterator
    for n_val in n_list:
        if n_val > num_nodes:
            continue
        
        # Generate combinations lazily using itertools.combinations
        hyperedges = list(itertools.combinations(range(num_nodes), n_val))
        
        # Chunk hyperedges into batches
        batch_size = 2500
        hyperedge_batches = [hyperedges[i:i + batch_size]
                             for i in range(0, len(hyperedges), batch_size)]
        
        # Process batches in parallel using Joblib
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_hyperedge_batch)(batch,
                                             old_x_shared,
                                             precomputed_bin_edges,
                                             precomputed_hist_values)
            for batch in hyperedge_batches
        )

        print(f"Finished processing {n_val}-point hyperedges.")

        # Flatten results from parallel execution
        for batch_result in results:
            for processed_hyperedge, eec_vals in batch_result:
                all_hyperedges.append(processed_hyperedge)
                all_hyperedge_attrs.append(eec_vals)

    total_hyperedges = len(all_hyperedges)
    
    # # Build the sparse incidence matrix.
    # row_indices = []
    # col_indices = []
    # for hyperedge_id, hyperedge in enumerate(all_hyperedges):
    #     for node in hyperedge:
    #         row_indices.append(node)
    #         col_indices.append(hyperedge_id)
    # 
    # indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    # values = torch.ones(indices.shape[1], dtype=torch.float32)
    # 
    # # Create a sparse COO tensor for the incidence matrix
    # hyperedge_index = torch.sparse_coo_tensor(indices, values,
    #                                           size=(num_nodes, total_hyperedges))
    
    # Build the hyperedge index as a [2, N] tensor.
    row_indices = []
    col_indices = []
    for hyperedge_id, hyperedge in enumerate(all_hyperedges):
        for node in hyperedge:
            row_indices.append(node)
            col_indices.append(hyperedge_id)
    hyperedge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)

    # Convert the list of hyperedge attributes to a tensor
    hyperedge_attr = torch.tensor(all_hyperedge_attrs, dtype=torch.float32)
    
    return hyperedge_index, hyperedge_attr


def get_eec_ls_values(data, N = 2, bins = 50, axis_range = (1e-3, 1)):
    """
    Get the EEC values for the given data.
    
    Parameters:
    data: np.ndarray
        The data for which the EEC values are to be calculated.
    N: int
        The number of nearest neighbors to consider.
    bins: int
        The number of bins to use for the histogram.
    axis_range: tuple
        The range of the x-axis.
        
    Returns:
    eec_ls: The EEC histogram with the bins and the values.
        The EEC values.
    """

    # Get the EEC values
    # Create an instance of the EECLongestSide class
    eec_ls = eec.EECLongestSideLog(N, bins, axis_range)

    # Multicore compute for EECLongestSide
    eec_ls(data)
    print(eec_ls)

    # Scaling eec values
    eec_ls.scale(1/eec_ls.sum())

    return eec_ls

def plot_eec_values(additional_edge_attrs, additional_hypergraph_attrs, eec_prop, output_dir='.'):
    """
    Plot the Energy-Energy Correlator (EEC) values against the RL bins.
    
    Parameters:
    additional_edge_attrs: list
        List of EEC objects for 2-point correlations
    additional_hypergraph_attrs: list
        List of EEC objects for n-point correlations (n > 2)
    eec_prop: list
        Properties used for EEC calculation [N, bins, (R_Lmin, R_Lmax)]
    output_dir: str
        Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    
    # Create output directory for plots if it doesn't exist
    plot_dir = os.path.join(output_dir, 'eec_plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Plot 2-point EEC if available
    if additional_edge_attrs:
        eec_obj = additional_edge_attrs[0]
        hist_values, y_errs = eec_obj.get_hist_errs(0, False)
        bin_edges = eec_obj.bin_edges()
        bin_centers = eec_obj.bin_centers()
        print(bin_edges, bin_centers, hist_values, y_errs)
        
        # plt.plot(bin_centers, hist_values, 'o-', linewidth=2, label='2-point EEC')
        plt.errorbar(
            bin_centers, hist_values,
            xerr=(bin_centers - bin_edges[:-1], bin_edges[1:] - bin_centers),
            yerr=y_errs,
            label='2-point EEC',
            fmt='s'
            )
    
    # Plot n-point EEC if available
    if additional_hypergraph_attrs:
        for i, eec_obj in enumerate(additional_hypergraph_attrs):
            n_point = eec_prop[0][i+1]  # +1 because we skip the 2-point in hypergraph
            hist_values, y_errs = eec_obj.get_hist_errs(0, False)
            bin_edges = eec_obj.bin_edges()
            bin_centers = eec_obj.bin_centers()
            
            # plt.plot(bin_centers, hist_values, 'o-', linewidth=2, label=f'{n_point}-point EEC')
            plt.errorbar(
                bin_centers, hist_values,
                xerr=(bin_centers - bin_edges[:-1], bin_edges[1:] - bin_centers),
                yerr=y_errs,
                label=f'{n_point}-point EEC',
                fmt='s'
                )

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-4, 1)
    plt.xlabel('RL (Angular Distance)', fontsize=14)
    plt.ylabel('EEC Value', fontsize=14)
    plt.title('Energy-Energy Correlators vs Angular Distance', fontsize=16)
    # plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    # Plot properties
    plt.annotate(f"Bins: {eec_prop[1]}, Range: {eec_prop[2]}",
                 xy=(0.02, 0.02), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eec_vs_rl.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, 'eec_vs_rl.pdf'))
    plt.close()
    
    print(f"EEC plot saved to {os.path.join(plot_dir, 'eec_vs_rl.png')}")

# function to one hot encode the jet type and leave the rest of the features as is
def OneHotEncodeType(x: np.ndarray):
    enc = OneHotEncoder(categories=[[0, 1, 2]]) # Assuming jet types are 0, 1, 2
    type_encoded = enc.fit_transform(x[..., 0].reshape(-1, 1)).toarray()
    other_features = x[..., 1:].reshape(-1, 3)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)

# @nb.njit(fastmath=True)
def normalize_array(arr, method='zscore', return_stats=False):
    """
    Normalize array using either z-score or min-max normalization.
    
    Note: Not JIT-compiled because it returns dictionaries with varying structures.
    
    Parameters:
    -----------
    arr : np.ndarray
        Array to normalize
    method : str, optional (default='zscore')
        Normalization method: 'zscore' or 'minmax'
        - 'zscore': (x - mean) / std
        - 'minmax': (x - min) / (max - min)
    return_stats : bool, optional (default=False)
        If True, return (normalized_arr, stats_dict)
        If False, return only normalized_arr
    
    Returns:
    --------
    normalized_arr : np.ndarray
        Normalized array
    stats_dict : dict (only if return_stats=True)
        Dictionary containing normalization statistics:
        - For 'zscore': {'mean': float, 'std': float, 'method': 'zscore'}
        - For 'minmax': {'min': float, 'max': float, 'method': 'minmax'}
    """
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean = np.nanmean(arr)
        std_dev = np.nanstd(arr)
        
        # Avoid division by zero
        if std_dev == 0:
            std_dev = 1.0
        
        normalized_arr = np.where(~np.isclose(arr, 0), (arr - mean) / std_dev, 0.0)
        normalized_arr = np.nan_to_num(normalized_arr, nan=0.0)
        
        stats = {'mean': float(mean), 'std': float(std_dev), 'method': 'zscore'}
        
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        
        # Avoid division by zero
        if arr_max - arr_min == 0:
            normalized_arr = np.zeros_like(arr)
        else:
            normalized_arr = (arr - arr_min) / (arr_max - arr_min)
            normalized_arr = np.nan_to_num(normalized_arr, nan=0.0)
        
        stats = {'min': float(arr_min), 'max': float(arr_max), 'method': 'minmax'}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'.")
    
    if return_stats:
        return normalized_arr, stats
    else:
        return normalized_arr


def reclusterJets(jet, R=0.4, pt_cut=0):
    """
    Recluster the jets.
    
    Parameters:
    jet: np.ndarray
        The jets to be reclusted.
    R: float
        The radius parameter.
    pt_cut: float
        The pt cut.
        
    Returns:
    reclustered_jets: np.ndarray
        The reclustered jets.
    """

    # Create a jet definition
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)

    # Create a cluster sequence
    cs = fastjet.ClusterSequence(jet, jet_def)

    return cs.constituents()[0], cs.inclusive_jets()

def plot_jet_kinematics(inclusive_jet, input_type=''):
    pt_list = []
    y_list = []
    phi_list = []
    
    for jet in inclusive_jet:

        if input_type=='hadronic':

            pt = jet[:, 0]
            y = jet[:, 1]
            phi = jet[:, 2]

            print(pt)
            # pt = jet[0]
            # y = jet[1]
            # phi = jet[2]

        else:
            # Extract E, px, py, pz
            E = jet[0]
            px = jet[1]
            py = jet[2]
            pz = jet[3]
            
            # Calculate pt, y, phi
            pt = np.sqrt(px**2 + py**2)
            y = 0.5 * np.log((E + pz) / (E - pz))
            phi = np.arctan2(py, px)
        
        # Append to lists
        pt_list.append(pt)
        y_list.append(y)
        phi_list.append(phi)
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot pt 
    axs[0].hist(pt_list, bins=50)
    axs[0].set_xlabel('pt')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('pt')

    # Plot y
    axs[1].hist(y_list, bins=50)
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('y')
    
    # Plot phi
    axs[2].hist(phi_list, bins=50)
    axs[2].set_xlabel('phi')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('phi')


    # # Plot pt vs y
    # axs[0].scatter(pt_list, y_list)
    # axs[0].set_xlabel('pt')
    # axs[0].set_ylabel('y')
    # axs[0].set_title('pt vs y')
    # 
    # # Plot y vs phi
    # axs[1].scatter(y_list, phi_list)
    # axs[1].set_xlabel('y')
    # axs[1].set_ylabel('phi')
    # axs[1].set_title('y vs phi')
    #
    # # Plot pt vs phi
    # axs[2].scatter(pt_list, phi_list)
    # axs[2].set_xlabel('pt')
    # axs[2].set_ylabel('phi')
    # axs[2].set_title('pt vs phi')

    # Save the plot
    plt.tight_layout()
    plt.savefig('kinematics_plot.png')
    

# BUG: The following function is not working as expected
def ms2pids(ms):
    """
    Convert the masses to pids.
    
    Parameters:
    ms: np.ndarray
        The masses to convert.
        
    Returns:
    pids: np.ndarray
        The pids.
    """

    pidsDict = {
    #   PDGID     CHARGE MASS          NAME
        0:       ( 0.,   0.,      ), # void
        1:       (-1./3, 0.33,    ), # down
        2:       ( 2./3, 0.33,    ), # up
        3:       (-1./3, 0.50,    ), # strange
        4:       ( 2./3, 1.50,    ), # charm
        5:       (-1./3, 4.80,    ), # bottom
        6:       ( 2./3, 171.,    ), # top
        11:      (-1.,   5.11e-4, ), # e-
        12:      ( 0.,   0.,      ), # nu_e
        13:      (-1.,   0.10566, ), # mu-
        14:      ( 0.,   0.,      ), # nu_mu
        15:      (-1.,   1.77682, ), # tau-
        16:      ( 0.,   0.,      ), # nu_tau
        21:      ( 0.,   0.,      ), # gluon
        22:      ( 0.,   0.,      ), # photon
        23:      ( 0.,   91.1876, ), # Z
        24:      ( 1.,   80.385,  ), # W+
        25:      ( 0.,   125.,    ), # Higgs
        111:     ( 0.,   0.13498, ), # pi0
        113:     ( 0.,   0.77549, ), # rho0
        130:     ( 0.,   0.49761, ), # K0-long
        211:     ( 1.,   0.13957, ), # pi+
        213:     ( 1.,   0.77549, ), # rho+
        221:     ( 0.,   0.54785, ), # eta
        223:     ( 0.,   0.78265, ), # omega
        310:     ( 0.,   0.49761, ), # K0-short
        321:     ( 1.,   0.49368, ), # K+
        331:     ( 0.,   0.95778, ), # eta'
        333:     ( 0.,   1.01946, ), # phi
        445:     ( 0.,   3.55620, ), # chi_2c
        555:     ( 0.,   9.91220, ), # chi_2b
        2101:    ( 1./3, 0.57933, ), # ud_0
        2112:    ( 0.,   0.93957, ), # neutron
        2203:    ( 4./3, 0.77133, ), # uu_1
        2212:    ( 1.,   0.93827, ), # proton
        1114:    (-1.,   1.232,   ), # Delta-
        2114:    ( 0.,   1.232,   ), # Delta0
        2214:    ( 1.,   1.232,   ), # Delta+
        2224:    ( 2.,   1.232,   ), # Delta++
        3122:    ( 0.,   1.11568, ), # Lambda0
        3222:    ( 1.,   1.18937, ), # Sigma+
        3212:    ( 0.,   1.19264, ), # Sigma0
        3112:    (-1.,   1.19745, ), # Sigma-
        3312:    (-1.,   1.32171, ), # Xi-
        3322:    ( 0.,   1.31486, ), # Xi0
        3334:    (-1.,   1.67245, ), # Omega-
        10441:   ( 0.,   3.41475, ), # chi_0c
        10551:   ( 0.,   9.85940, ), # chi_0b
        20443:   ( 0.,   3.51066, ), # chi_1c
        9940003: ( 0.,   3.29692, ), # J/psi[3S1(8)]
        9940005: ( 0.,   3.75620, ), # chi_2c[3S1(8)]
        9940011: ( 0.,   3.61475, ), # chi_0c[3S1(8)]
        9940023: ( 0.,   3.71066, ), # chi_1c[3S1(8)]
        9940103: ( 0.,   3.88611, ), # psi(2S)[3S1(8)]
        9941003: ( 0.,   3.29692, ), # J/psi[1S0(8)]
        9942003: ( 0.,   3.29692, ), # J/psi[3PJ(8)]
        9942033: ( 0.,   3.97315, ), # psi(3770)[3PJ(8)]
        9950203: ( 0.,   10.5552, ), # Upsilon(3S)[3S1(8)]
    }

    particleMassesDict  = {pdgid: props[1] for pdgid,props in pidsDict.items()}

    pids = []
    for m in ms:
        for pdgid, mass in particleMassesDict.items():
            print(mass, m)
            if np.isclose(mass, m):
                pids.append(pdgid)

    pids = np.array(pids)

    return pids
