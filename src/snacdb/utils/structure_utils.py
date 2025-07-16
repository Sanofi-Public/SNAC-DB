from collections.abc import Iterable
import scipy
import itertools
import networkx as nx
import numpy as np
from itertools import chain

def get_contact_map(coord_chain1, coord_chain2, cutoff=8.0):
    """
    Computes a binary contact map between two chains based on a distance cutoff.

    Parameters:
    - coord_chain1: NumPy array of coordinates for the first chain.
    - coord_chain2: NumPy array of coordinates for the second chain.
    - cutoff: Distance threshold (in Angstroms) to define a contact.

    Returns:
    - A binary matrix indicating contacts (1 if in contact, 0 otherwise).
    """
    dist_matrix = scipy.spatial.distance.cdist(coord_chain1, coord_chain2)
    contact_map = np.where(dist_matrix < cutoff, 1, 0)
    return contact_map

def any_in(a, b):
    """
    Checks if any element in list `a` is present in list `b`.

    Parameters:
    - a: Single element or iterable.
    - b: Single element or iterable.

    Returns:
    - True if at least one element of `a` is found in `b`, otherwise False.
    """
    if not isinstance(a, list):
        if isinstance(a, Iterable) and not isinstance(a, str):
            a = list(a)
        else:
            a = [a]
    if not isinstance(b, list):
        if isinstance(b, Iterable) and not isinstance(b, str):
            b = list(b)
        else:
            b = [b]
    
    return bool(set(a) & set(b))  # Returns True if there is any intersection

def compute_total_contacts_between_pair_of_chain_tuples(structure, chain_tup1, chain_tup2, get_full_map=False, cutoff=8.0):
    """
    Computes the total number of atomic contacts between two chain tuples.

    Parameters:
    - structure: Dictionary containing per-chain atom coordinates.
    - chain_tup1: A tuple of chain identifiers for the first group.
    - chain_tup2: A tuple of chain identifiers for the second group.
    - get_full_map: If True, returns the full contact map.

    Returns:
    - Total count of nonzero contacts.
    - If get_full_map is True, also returns the contact map.
    """
    if isinstance(chain_tup1, str):
        chain_tup1 = (chain_tup1,)
    if isinstance(chain_tup2, str):
        chain_tup2 = (chain_tup2,)

    # Extract C-alpha (Cα) coordinates from each chain
    coord_chain_tup1 = [structure[chain]['atom37'][:, 1, :] for chain in chain_tup1]
    coord_chain_tup2 = [structure[chain]['atom37'][:, 1, :] for chain in chain_tup2]

    # Concatenate to form a single array for each chain group
    coord_chain_tup1 = np.concatenate(coord_chain_tup1)
    coord_chain_tup2 = np.concatenate(coord_chain_tup2)
    
    # Compute contact map
    contact_map = get_contact_map(coord_chain_tup1, coord_chain_tup2, cutoff=cutoff)
    
    # Count the number of valid contacts (excluding NaNs)
    total_contacts = np.count_nonzero(~np.isnan(contact_map) & (contact_map != 0))

    if get_full_map:
        return total_contacts, contact_map
    return total_contacts

def get_disjoint_subgraphs(node_list, structure, prefix='', return_graph=False, cutoff=8.0):
    """
    Identifies disjoint subgraphs of chains based on contact connectivity.

    Parameters:
    - node_list: List of chain identifiers.
    - structure: Dictionary containing chain coordinates.
    - prefix: String prefix for node names in the graph.
    - return_graph: If True, returns the full NetworkX graph.

    Returns:
    - A list of disjoint chain groups (as tuples).
    - If return_graph is True, also returns the graph and node dictionary.
    """
    # Create a mapping from prefixed node names to chain identifiers
    node_dict = {f'{prefix}_{i}': node for i, node in enumerate(node_list)}

    # Create an undirected graph
    G = nx.Graph()
    G.add_nodes_from(node_dict)

    # Generate all possible chain pairs
    edges_pairs = list(itertools.combinations(node_dict, 2))
    # Identify contacting chain pairs
    edges = []
    for edge in edges_pairs:
        chain1 = node_dict[edge[0]]
        chain2 = node_dict[edge[1]]
        total_contacts = compute_total_contacts_between_pair_of_chain_tuples(structure, chain1, chain2, cutoff=cutoff)
        if total_contacts > 0:
            edges.append(edge)

    # Add edges representing interacting chain pairs
    G.add_edges_from(edges)

    # Identify connected components (disjoint sets of interacting chains)
    set_of_disjoint_nodes = [tuple(c) for c in nx.connected_components(G)]

    # Convert back to original chain identifiers
    set_of_disjoint_nodes_converted_to_chains = [
        tuple(node_dict[v] for v in val) for val in set_of_disjoint_nodes
    ]

    if return_graph:
        return set_of_disjoint_nodes_converted_to_chains, G, node_dict
    return set_of_disjoint_nodes_converted_to_chains

def identify_VH_VL_pairs_in_contact(chain_dict, return_VHH_chains=False):
    """
    Identifies VH-VL pairs that are in contact, filtering out VH-VH or VHH-VL interactions.

    Parameters:
    - chain_dict: Dictionary with separate 'VH' and 'VL' chain groups.

    Returns:
    - A list of VH-VL chain pairs in contact.
    """
    if 'VL' not in chain_dict:
        print('No light chain to compare with!')
        return None

    # Flatten the chain dictionary for easy access
    chain_dict_flat = {key: value for group in ['VH', 'VL'] for key, value in chain_dict[group].items()}

    VH_keys = list(chain_dict['VH'].keys())
    VL_keys = list(chain_dict['VL'].keys())    
    VH_VL_edges = []

    for cutoff in (8.0, 10.0, 12.0):
        VH_leftover = list(set(VH_keys) - set([c[0] for c in VH_VL_edges]))
        VL_leftover = list(set(VL_keys) - set([c[1] for c in VH_VL_edges]))
        if len(VH_leftover) >= 0 and len(VL_leftover) == 0:
            break
        VH_VL_edges += identify_VH_VL_pairs_in_contact_relax_criteria(VH_leftover, VL_leftover, chain_dict_flat, cutoff)

    VH_leftover = list(set(VH_keys) - set([c[0] for c in VH_VL_edges]))
    VL_leftover = list(set(VL_keys) - set([c[1] for c in VH_VL_edges]))
    if len(VL_leftover) != 0:
        print("Error: unpaired light chain. Please make sure to manually review this structure")
        #raise ValueError("There is an unpair light chain. Please make sure to manually review this structure.")

    if return_VHH_chains:
        VHH_chains = set(chain_dict['VH'].keys()) - {vh for vh, vl in VH_VL_edges}
        return VH_VL_edges, list(VHH_chains)
    else:
        return VH_VL_edges

def identify_VH_VL_pairs_in_contact_relax_criteria(VH_keys, VL_keys, chain_dict_flat, cutoff):
    """
    Applies a strict criteria for determining VH-VL pairs at a specified cutoff distance

    Parameters:
    - VH_keys (list): List of chain IDs for heavy chains.
    - VL_keys (list): List of chain IDs for light chains.
    - chain_dict_flat (dict): dictionary detailing the structure
    - cutoff (float): distance cutoff value in angstrom

    Returns:
    - A list of VH-VL chain pairs in contact.
    """
    
    disjoint_chains, G, node_dict = get_disjoint_subgraphs(VH_keys + VL_keys, chain_dict_flat, return_graph=True, cutoff=cutoff)
    
    # Extract edges representing chain contacts
    edges = [(node_dict[e[0]], node_dict[e[1]]) for e in G.edges]

    VH_VL_edges_check = {}
    for edge in edges:
        # Check if this edge represents a VH-VL interaction
        if any_in(VH_keys, edge) and any_in(VL_keys, edge):
            if any_in(VH_keys, [edge[1]]):
                edge = (edge[1], edge[0])  # Ensure VH is first in the tuple

            # Check if VH-VL pair has contact along framework regions
            _, contact_map = compute_total_contacts_between_pair_of_chain_tuples(chain_dict_flat, edge[0], edge[1], get_full_map=True, cutoff=cutoff)

            imgt_num_heavy = []
            imgt_num_light = []
            imgt_residues = chain(range(1, 8), range(39, 55), range(66,104), range(105, 117), range(118, 124))
            # check all the resiudes for contacts between the heavy and light
            for idx in imgt_residues:  # section of FR1
                if str(idx) in chain_dict_flat[edge[0]]['imgt']:
                    imgt_num_heavy.append(chain_dict_flat[edge[0]]['imgt'].index(str(idx)))
                if str(idx) in chain_dict_flat[edge[1]]['imgt']:
                    imgt_num_light.append(chain_dict_flat[edge[1]]['imgt'].index(str(idx)))

            # determine contact map at the specified indices
            interaction = contact_map[np.ix_(imgt_num_heavy, imgt_num_light)]
            total_contacts = np.count_nonzero(~np.isnan(interaction) & (interaction))
            # check to see if enough contacts are made to determine a pair
            if total_contacts >= 25:
                VH_VL_edges_check[edge] = total_contacts
            if len(imgt_num_heavy) == 0 or len(imgt_num_light) == 0:
                print(f'Could not find position between any of the framework region in {edge[0]} sequence.')

        # need to check that all the complexes have unique pairs
        skip = []
        for pair_1 in VH_VL_edges_check.copy():
            if pair_1 in skip:
                continue
            for pair_2 in VH_VL_edges_check.copy():
                if pair_2 in skip:
                    continue
                if pair_1 != pair_2:
                    # check if pairs have the same chain ID
                    if len(set(pair_1).intersection(set(pair_2))) > 0:
                        if VH_VL_edges_check[pair_1] > VH_VL_edges_check[pair_2]: 
                            del VH_VL_edges_check[pair_2]
                            skip.append(pair_2)
                        # if they are equal raise a value error
                        elif VH_VL_edges_check[pair_1] == VH_VL_edges_check[pair_2]:
                            if VH_VL_edges_check[pair_1] < 25:
                                del VH_VL_edges_check[pair_1]
                                del VH_VL_edges_check[pair_2]
                                skip += [pair_1, pair_2]
                            print("Error is occuring between VH-VL pairs:", pair_1, pair_2)
                            print("They have a total number of contacts being:", VH_VL_edges_check[pair_1])
                            raise ValueError
                        else:
                            del VH_VL_edges_check[pair_1]
                            skip.append(pair_1)

    return list(VH_VL_edges_check.keys())