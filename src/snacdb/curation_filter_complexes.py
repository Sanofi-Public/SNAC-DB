import pandas as pd
import sys
import os
import ast
import numpy as np
import itertools
import argparse
import shutil
from snacdb.utils.pdb_utils_clean_parse import PDBUtils, dict_to_structure
from snacdb.utils.sequence_utils import chain_dict_to_VH_VL_Ag_categories
from snacdb.utils.structure_utils import *
from snacdb.utils.parallelize import ParallelProcessorForCPUBoundTasks


def nan_any(contact_map, axis=-1):
    """
    Performs np.any while preserving NaN values.

    Args:
        contact_map (np.ndarray): Input boolean/numeric array.
        axis (int): Axis along which to apply np.any.

    Returns:
        np.ndarray: Result after applying np.any with NaN preservation.
    """
    nan_mask = np.isnan(contact_map)  # Identify NaNs
    result = np.any(contact_map==1, axis=axis)  # Apply np.any
    
    # If all values along the axis were NaN, keep NaN in result
    all_nan_mask = np.all(nan_mask, axis=axis)
    result = result.astype(float)  # Convert to float to accommodate NaNs
    result[all_nan_mask] = np.nan  # Assign NaN where all values were NaN
    
    return result


def get_contact_map_atom37(coord_chain1, coord_chain2, cutoff=8.0, alpha_carbon=True):
    """
    Determines contact map between two different chains with a specified cutoff
     distance. As well, you can specify to do the contact map using all the heavy
     atoms from atom37 coordinate space or to just use alpha carbons. 

    Args:
        coord_chain1 (np.ndarray): Atom37 coordinate for structure 1.
        coord_chain2 (np.ndarray): Atom37 coordinate for structure 2.
        cutoff (float): The maximum distance where contact is considered
        alpha_carbon (bool): Whether to calculate the contact map for only the 
         alpha carbon of the structure

    Returns:
        contact_map (np.ndarray): Matrix showing contact between structure 1 
         and 2.
    """

    if alpha_carbon: # only considering alpha carbons for atom37 structures
        coord_chain1 = coord_chain1[:, 1:2, :]
        coord_chain2 = coord_chain2[:, 1:2, :]

    # distance matrix of the two structures
    dist_mat = np.sqrt(np.sum((coord_chain1[:, None, :, None, :] - coord_chain2[None, :, None, :, :])**2, axis=-1))
    # Determining contact within cutoff distance and reshaping matrix
    contact_map = np.where(np.isnan(dist_mat), np.nan, np.where(dist_mat < cutoff, 1, 0)) 
    contact_map = contact_map.reshape(list(contact_map.shape[:2]) + [-1])
    contact_map = nan_any(contact_map, axis=-1)
    return contact_map


def check_cdr_contact(V_chain_coords, target_chain_coords, imgt, cutoff=8.0):
    """
    Determines whether there is contact in the CDR region between 
     V_chain_coords and target_chain_coords.

    Args:
        V_chain_coords (np.ndarray): Atom37 coordinate for structure 1.
        target_chain_coords (np.ndarray): Atom37 coordinate for antigen.
        imgt (list): The imgt numbering.
        cutoff (float): The maximum distance where contact is considered.

    Returns:
        contact_dict (dict): Dictionary containing information about if there is contact
        with each CDR loop.
    """

    # Check if the contact happens in the CDR regions
    contact_dict = {'CDR3' : False, 'CDR2' : False, 'CDR1' : False}
    cdrs = [range(27, 38), range(56, 65), range(105, 117)]
    # determining if there is contact for each region in the CDR loop
    for i, region in enumerate(contact_dict.keys()):

        imgt_num = []
        for idx in cdrs[i]:  # section of FR1
            if str(idx) in imgt:
                imgt_num.append(imgt.index(str(idx)))

        contact_map = get_contact_map_atom37(V_chain_coords, target_chain_coords, cutoff=cutoff)
        
        # total_contacts = np.nansum(contact_map[crd_lims[0]: crd_lims[1], :])
        interaction = contact_map[imgt_num, :]
        total_contacts = np.nansum(interaction)

        if total_contacts > 0:
            contact_dict[region] = True

    return contact_dict


def has_common_substring(seq1, seq2, perc=0.98):
    """
    Determines if seq1 is located in seq2 by a specified percent,
    or vice-versa.

    Args:
        seq1 (str): Sequence for chain 1.
        seq2 (str): Sequence for chain 2.
        perc (float): The percentage criteria for the two sequences
         to match to. Ranges from 0 to 1

    Returns:
        bool: If one of the sequences is inside the other.
    """
    # determining the shorter sequence
    shorter_seq = seq1 if len(seq1) < len(seq2) else seq2
    longer_seq = seq2 if len(seq1) < len(seq2) else seq1

    # determine if shorter_seq in longer_seq
    min_length = int(perc * len(shorter_seq))
    for i in range(len(shorter_seq) - min_length + 1):
        substring = shorter_seq[i:i+min_length]
        if substring in longer_seq:
            return True

    return False


def check_for_stringent_condition(Ab, Ag, chain_dict, no_cdr1=False, cutoff=8.0):
    """
    Determines the number of contacts that occur between Ab and Ag.

    Args:
        Ab (tuple/str): Chain IDs of VH/VL/VHH.
        Ag (str): Chain ID of the antigen of interest.
        chain_dict (dict): Dictionary containing information on the
         structure.
        no_cdr1 (bool): Whether to not account for the CDR1 region
        cutoff (float): The maximum cutoff distance (in terms of angstroms) for
         the contact map

    Returns:
        int: The count of how many contacts are recorded between Ab and Ag.
    """
    # determining if Ab or sdAb
    if type(Ab) != tuple:
        Ab = tuple(Ab)

    contacts = []
    for chain in Ab:  # determining contact between antigen and antibody
        Ab_dict = chain_dict[chain]
        Ag_atom37 = chain_dict[Ag]['atom37']
        cdr_contact = check_cdr_contact(Ab_dict['atom37'], Ag_atom37, Ab_dict['imgt'], cutoff)
        if no_cdr1:  # ignoring contact from CDR1
            del cdr_contact["CDR1"]
        contacts += [contact if contact is not None else 0 for contact in cdr_contact.values()]

    return sum(contact > 0 for contact in contacts)


def filter_func_VH_VL(VH_VL, target, chain_dict):
    """
    This filter function is geared towards Antibodies. The conditions for contact
     is different for different antigens. For any antigen that is not a VH-VL pair
     then contact is determine if there is at least heavy chain interaction, or at
     least one heavy chain and one light chain interaction for any of the CDR loops.
     If the antigen is a VH-VL pair that is identical to the antibody then a contact
     is determine only if at least one heavy chain and one light chain interaction
     is observed, and this is only factoring CDR3 and CDR2 interactions.

    Args:
        VH_VL (list): List of Heavy Chain and Light Chain ID.
        target (list): List of Antigen chain IDs.
        chain_dict (dict): Dictionary containing information on the structure.

    Returns:
        chain_dict_new (dict): Contains information about the complex and
         its labeling.
        comments (str): Contains information about how the complex is created.
    """

    # comments will describe changes that occur to complex over the function
    comments = {"Complex_Identity": [], "Single-Chain_Antigens": [], "Multi-Chain_Antigen": [], "Secondary_Check": [], "Replicate_Chains": [], "Complex_Summary": []}
    chains_checked_in_round_1_filtering_VH_VL = []

    # determining if any of the antigen has replicates
    seq_VH = chain_dict[VH_VL[0]]["seq"]  # sequence of heavy chain
    seq_VL = chain_dict[VH_VL[1]]["seq"]  # sequence of light chain
    seq_target = [chain_dict[c]['seq'] for c in target]
    test_VH = any([has_common_substring(seq_VH, seq) for seq in seq_target])
    test_VL = any([has_common_substring(seq_VL, seq) for seq in seq_target])
    original_target = target.copy()

    if len(target) != 0:  # complex has an antigen chain
        # if test_VH or test_VL:  # a chain matches either the VH or VL
        already_added = []
        VH_VL_pairs_in_target = []

        # determining which antigens are multi-chain antigens
        for c in target:
            if "pair" in chain_dict[c].keys():
                if c not in already_added:
                    already_added.append(chain_dict[c]["pair"])
                    VH_VL_pairs_in_target.append((c, chain_dict[c]["pair"]))

        # Solves for antigens that are not VH-VL
        chains_in_target_part_of_VH_VL_pairs = list(set(list(itertools.chain.from_iterable(VH_VL_pairs_in_target))))
        other_chains_in_target = [c for c in target if not any_in(c, chains_in_target_part_of_VH_VL_pairs)] 

        target_updated = []
        chains_checked = []
        other_chains_removed = []
        # determines if there is contact between the antigen (not VH-VL) and antibody
        for c in other_chains_in_target:
            chains_checked.append(c)
            test_VH = has_common_substring(seq_VH, chain_dict[c]['seq'])
            test_VL = has_common_substring(seq_VL, chain_dict[c]['seq'])
            if test_VH or test_VL:  # single-chain replicate
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c, chain_dict, no_cdr1=True)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c, chain_dict, no_cdr1=True)
                # determining if contact is sufficent for interaction
                if (num_contacts_greater_than_zero_VH >= 1 or num_contacts_greater_than_zero_VL >= 1):
                    chain_dict[c]["interaction_type"] = "Weak"
                    target_updated.append(c)
                    comments["Replicate_Chains"].append(c)
                    comments["Single-Chain_Antigens"].append(f'Replicate Chain {c} added to complex and has {num_contacts_greater_than_zero_VH} Heavy Chain Contacts and {num_contacts_greater_than_zero_VL} Light Chain Contacts')
                else:
                    other_chains_removed.append(c)
                    comments["Single-Chain_Antigens"].append(f'V-domain chain {c} removed from complex because of not enough CDR contacts ({num_contacts_greater_than_zero_VH}, {num_contacts_greater_than_zero_VL})')
            else:  # doing check against single-chain antigen
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c, chain_dict)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c, chain_dict)
                if (num_contacts_greater_than_zero_VH >= 1 or num_contacts_greater_than_zero_VL >= 1):
                    chain_dict[c]["interaction_type"] = "Strong"
                    target_updated.append(c)
                    comments["Single-Chain_Antigens"].append(f'Chain {c} added to complex and has {num_contacts_greater_than_zero_VH} Heavy Chain Contacts and {num_contacts_greater_than_zero_VL} Light Chain Contacts')
                else:
                    other_chains_removed.append(c)
                    comments["Single-Chain_Antigens"].append(f'V-domain chain {c} removed from complex because of not enough CDR contacts')

        VH_VL_pairs_in_target_removed = []
        # testing multi-chain antigens for interaction
        for c in VH_VL_pairs_in_target:
            seq_VH_VL_pairs_in_target = [chain_dict[c[0]]['seq'], chain_dict[c[1]]['seq']]
            test_VH = any([has_common_substring(seq_VH, seq) for seq in seq_VH_VL_pairs_in_target])
            test_VL = any([has_common_substring(seq_VL, seq) for seq in seq_VH_VL_pairs_in_target])
            chains_checked += list(c)
            if test_VH or test_VL:  # multi-chain antigen is a replicate
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c[0], chain_dict, no_cdr1=True) + check_for_stringent_condition(VH_VL[0], c[1], chain_dict, no_cdr1=True)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c[0], chain_dict, no_cdr1=True) + check_for_stringent_condition(VH_VL[1], c[1], chain_dict, no_cdr1=True)
                # testing if contact is enough for interaction
                if num_contacts_greater_than_zero_VH >= 1 and num_contacts_greater_than_zero_VL >= 1:
                    chain_dict[c[0]]["interaction_type"] = "Weak"
                    chain_dict[c[1]]["interaction_type"] = "Weak"
                    target_updated += list(c)
                    comments["Replicate_Chains"].append(c[0])
                    comments["Replicate_Chains"].append(c[1])
                    comments["Multi-Chain_Antigen"].append(f'Replicate pair {c} added to complex and has {num_contacts_greater_than_zero_VH} Heavy Chain Contacts and {num_contacts_greater_than_zero_VL} Light Chain Contacts')
                else:
                    VH_VL_pairs_in_target_removed.append(c)
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} removed from complex because of not enough CDR contacts ({num_contacts_greater_than_zero_VH},{num_contacts_greater_than_zero_VL})')

            else:  # doing check against non-replicate multi-chain antigen 
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c[0], chain_dict) + check_for_stringent_condition(VH_VL[0], c[1], chain_dict)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c[0], chain_dict) + check_for_stringent_condition(VH_VL[1], c[1], chain_dict)
                if (num_contacts_greater_than_zero_VH >= 1 and num_contacts_greater_than_zero_VL >= 1):
                    chain_dict[c[0]]["interaction_type"] = "Strong"
                    chain_dict[c[1]]["interaction_type"] = "Strong"
                    target_updated += list(c)
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} added to complex and has {num_contacts_greater_than_zero_VH} Heavy Chain Contacts and {num_contacts_greater_than_zero_VL} Light Chain Contacts')
    
                else:
                    VH_VL_pairs_in_target_removed.append(c)
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} removed from complex because of not enough CDR contacts')

        # updating the target after round 1 filtration
        target_updated = tuple(target_updated)
        VH_VL_complex_filtered = [VH_VL, target_updated]
        if len(chains_checked) > 0:
            chains_checked_in_round_1_filtering_VH_VL = chains_checked

    else:  # no filtering needed if no antigen chain exists
        VH_VL_complex_filtered = [VH_VL, target]
        
    # starting round 2 of filtration
    VH_VL, target = VH_VL_complex_filtered

    # doing secondary check only if no antigen chains made it past the curation step
    if len(target) == 0 and len(original_target) != 0:
        target_updated = []

        # check all single chain antigens
        for c in other_chains_in_target:
            test_VH = has_common_substring(seq_VH, chain_dict[c]['seq'])
            test_VL = has_common_substring(seq_VL, chain_dict[c]['seq'])
            if test_VH or test_VL: # if single chain is a replicate
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c, chain_dict, cutoff=12.0, no_cdr1=True)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c, chain_dict, cutoff=12.0, no_cdr1=True)
                # test if contact is enough to determine interaction
                if (num_contacts_greater_than_zero_VH >= 1 or num_contacts_greater_than_zero_VL >= 1):
                    chain_dict[c]["interaction_type"] = "Weak"
                    target_updated.append(c)
                    comments["Replicate_Chains"].append(c)
                    comments["Secondary_Check"].append(f'Chain {c} added to complex and has {num_contacts_greater_than_zero_VH} Heavy Chain Contacts and {num_contacts_greater_than_zero_VL} Light Chain Contacts after the cutoff being changed to 12 Angstroms')
                else:
                    comments["Secondary_Check"].append(f'Chain {c} was not added to complex due to few interaction ({num_contacts_greater_than_zero_VH}, {num_contacts_greater_than_zero_VL})')
            else:  # automatically add antigen chains to complex
                chain_dict[c]["interaction_type"] = "Strong"
                target_updated.append(c)
                comments["Secondary_Check"].append(f"Adding back chain {c} due to no chain passing the filtering step")

        # check multichain antigen chains
        for c in VH_VL_pairs_in_target:
            seq_VH_VL_pairs_in_target = [chain_dict[c[0]]['seq'], chain_dict[c[1]]['seq']]
            test_VH = any([has_common_substring(seq_VH, seq) for seq in seq_VH_VL_pairs_in_target])
            test_VL = any([has_common_substring(seq_VL, seq) for seq in seq_VH_VL_pairs_in_target])
            chains_checked += list(c)

            if test_VH or test_VL:  # antigen chain is a replicate
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c[0], chain_dict, cutoff=12.0) + check_for_stringent_condition(VH_VL[0], c[1], chain_dict, cutoff=12.0, no_cdr1=True)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c[0], chain_dict, cutoff=12.0) + check_for_stringent_condition(VH_VL[1], c[1], chain_dict, cutoff=12.0, no_cdr1=True)
                # testing if contact is sufficient for interaction
                if (num_contacts_greater_than_zero_VH >= 1 and num_contacts_greater_than_zero_VL >= 1):
                    chain_dict[c[0]]["interaction_type"] = "Weak"
                    chain_dict[c[1]]["interaction_type"] = "Weak"
                    target_updated += list(c)
                    comments["Replicate_Chains"].append(c[0])
                    comments["Replicate_Chains"].append(c[1])
                    comments["Secondary_Check"].append(f'Chain pair {c} added to complex and has {num_contacts_greater_than_zero_VH} Heavy Chain Contacts and {num_contacts_greater_than_zero_VL} Light Chain Contacts after the cutoff distance was extended to 12 Angstroms')
                else:
                    comments["Secondary_Check"].append(f"Chain pair {c} not added to complex due to failing criteria ({num_contacts_greater_than_zero_VH}, {num_contacts_greater_than_zero_VL})")
            else:  # automatically add antigen chains back to complex
                num_contacts_greater_than_zero_VH = check_for_stringent_condition(VH_VL[0], c[0], chain_dict, cutoff=12.0) + check_for_stringent_condition(VH_VL[0], c[1], chain_dict, cutoff=12.0)
                num_contacts_greater_than_zero_VL = check_for_stringent_condition(VH_VL[1], c[0], chain_dict, cutoff=12.0) + check_for_stringent_condition(VH_VL[1], c[1], chain_dict, cutoff=12.0)
                if (num_contacts_greater_than_zero_VH >= 1 and num_contacts_greater_than_zero_VL >= 1):
                    chain_dict[c[0]]["interaction_type"] = "Strong"
                    chain_dict[c[1]]["interaction_type"] = "Strong"
                    target_updated += list(c)
                    comments["Secondary_Check"].append(f"Adding back chain pair {c} due to no antigen chain passing the filtering process")
                else:
                    comments["Secondary_Check"].append(f"Chain pair {c} not added to complex due to failing criteria ({num_contacts_greater_than_zero_VH}, {num_contacts_greater_than_zero_VL})")
        
        VH_VL_complex_filtered_round_2 = [VH_VL, target_updated]
    else:
        VH_VL_complex_filtered_round_2 = [VH_VL, list(target)]

    VH_VL, target = VH_VL_complex_filtered_round_2
    if len(target) > 0:
        # summarize changes that occured to the complex
        if len(original_target) != len(target):
            comments["Complex_Summary"].append(f'Updated complex by removing chains {set(original_target)-set(target)}. New Complex consists of antibodies: {VH_VL} and antigen chains: {target}')
        else:
            comments["Complex_Summary"].append(f'Complex remains unchanged after the filter process. Complex consists of antibodies: {VH_VL} and antigen chains: {target}')
    else:
        # detail how no antigens are in the complex
        if len(original_target) == 0:
            comments["Complex_Summary"].append(f'There were no antigen chains associated with the structure {VH_VL} before filter check')
        else:
            comments["Complex_Summary"].append(f'There are no antigen chains associated with the structure {VH_VL} after filter check possibly due to potential symmetry issue and not enough CDR contacts')

    # updating chain_dict with the new complex
    chain_dict_new = {chain:chain_dict[chain] for chain in (VH_VL + target)}
    return chain_dict_new, comments


def filter_func_VHH(VHH, target, chain_dict):
    """
    This filter function is geared towards Nanobodies. The conditions for being
    included in the complex is if there is contact between both the CDR3 and
    CDR2 region of the VHH and the antigen, at least with 16 angstrom distance.

    Args:
        VHH (str): Heavy Chain Chain ID.
        target (list): List of Antigen chain IDs.
        chain_dict (dict): Dictionary containing information on the structure.

    Returns:
        chain_dict_new (dict): Contains information about the complex and
         its labeling.
        comments (str): Contains information about how the complex is created.
    """

    # comments will describe changes that occur to complex over the function
    comments = {"Complex_Identity": [], "Single-Chain_Antigens": [], "Multi-Chain_Antigen": [], "Secondary_Check": [], "Replicate_Chains": [], "Complex_Summary": []}
    seq_VHH = chain_dict[VHH]['seq']  # sequence of VHH
    seq_target = [chain_dict[c]['seq'] for c in target]  # sequence of Ags
    original_target = target.copy()

    test_VHH = any([has_common_substring(seq_VHH, seq) for seq in seq_target])
    if len(target) != 0:  # complex has at least one antigen chain 
        already_added = []
        VH_VL_pairs_in_target = []
        # determining which antigens are multi-chain antigens
        for c in target:  # determine which antigens are VH-VL
            if "pair" in chain_dict[c].keys():
                if not (c in already_added):
                    already_added.append(chain_dict[c]["pair"])
                    VH_VL_pairs_in_target.append((c, chain_dict[c]["pair"]))

        # determining all Ag that are not VH-VL
        chains_in_target_part_of_VH_VL_pairs = list(set(list(itertools.chain.from_iterable(VH_VL_pairs_in_target))))
        other_chains_in_target = [c for c in target if not any_in(c, chains_in_target_part_of_VH_VL_pairs)] 

        target_updated = []
        chains_checked = []
        chains_removed = []

        # determining if the antigen is involved in the complex
        for c in other_chains_in_target:
            test_VHH = has_common_substring(seq_VHH, chain_dict[c]['seq'])
            chains_checked.append(c)
            if test_VHH:  # single-chain replicate
                num_contacts_greater_than_zero = check_for_stringent_condition(VHH, c, chain_dict)
                # testing to see if contact meets criteria of interaction
                if num_contacts_greater_than_zero >= 2:
                    chain_dict[c]["interaction_type"] = "Weak"
                    target_updated.append(c)
                    comments["Replicate_Chains"].append(c)
                    comments["Single-Chain_Antigens"].append(f'Chain {c} added to complex and has {num_contacts_greater_than_zero} Heavy Chain Contacts')

                else:
                    chains_removed.append(c)
                    comments["Single-Chain_Antigens"].append(f'V-domain chain {c} removed from complex because of not enough CDR contacts')
            else:  # single-chain non replicate
                num_contacts_greater_than_zero = check_for_stringent_condition(VHH, c, chain_dict)
                if num_contacts_greater_than_zero >= 1:
                    chain_dict[c]["interaction_type"] = "Strong"
                    target_updated.append(c)
                    comments["Single-Chain_Antigens"].append(f'Chain {c} added to complex and has {num_contacts_greater_than_zero} Heavy Chain Contacts')

                else:
                    chains_removed.append(c)
                    comments["Single-Chain_Antigens"].append(f'V-domain chain {c} removed from complex because of not enough CDR contacts')

        # determine if multi-chain antigen is in the complex
        VH_VL_pairs_in_target_removed = []
        for c in VH_VL_pairs_in_target:
            chains_checked += list(c)
            seq_VH_VL_pairs_in_target = [chain_dict[c[0]]['seq'], chain_dict[c[1]]['seq']]
            test_VHH = any([has_common_substring(seq_VHH, seq) for seq in seq_VH_VL_pairs_in_target])
            if test_VHH:  # multi-chain replicate
                num_contacts_greater_than_zero_VHH = check_for_stringent_condition(VHH, c[0], chain_dict, no_cdr1=True) + check_for_stringent_condition(VHH, c[1], chain_dict, no_cdr1=True)
                # testing for interaction
                if num_contacts_greater_than_zero_VHH >= 2:
                    chain_dict[c[0]]["interaction_type"] = "Weak"
                    chain_dict[c[1]]["interaction_type"] = "Weak"
                    target_updated += list(c)
                    comments["Replicate_Chains"].append(c[0])
                    comments["Replicate_Chains"].append(c[1])
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} added to complex and has {num_contacts_greater_than_zero_VHH} Heavy Chain Contacts')
                else:
                    VH_VL_pairs_in_target_removed.append(c)
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} removed from complex because of not enough CDR contacts')
            else:  # multi-chain non-replicate
                num_contacts_greater_than_zero_VHH = check_for_stringent_condition(VHH, c[0], chain_dict, no_cdr1=True) + check_for_stringent_condition(VHH, c[1], chain_dict, no_cdr1=True)
                if num_contacts_greater_than_zero_VHH >= 1:
                    chain_dict[c[0]]["interaction_type"] = "Strong"
                    chain_dict[c[1]]["interaction_type"] = "Strong"
                    target_updated += list(c)
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} added to complex and has {num_contacts_greater_than_zero_VHH} Heavy Chain Contacts')
                else:
                    VH_VL_pairs_in_target_removed.append(c)
                    comments["Multi-Chain_Antigen"].append(f'VH-VL pair {c} removed from complex because of not enough CDR contacts')

        # updating the final antigens that are still in complex
        target_updated = tuple(target_updated)
        if len(target_updated) > 0:
            if sorted(target) != sorted(target_updated):
                comments["Complex_Summary"].append(f'Updated complex by removing V-domain chains {chains_removed} because of potential symmetry issue and not enough CDR contacts')
        else:
            comments["Complex_Summary"].append(f'There are no antigen chains associated with the structure {VHH} after 1st check possibly due to potential symmetry issue and not enough CDR contacts')
        VHH_complex_filtered = [VHH, target_updated]
        if len(chains_checked) > 0:
            chains_checked_in_round_1_filtering_VHH = chains_checked
    else:
        VHH_complex_filtered = [VHH, target]
        chains_checked_in_round_1_filtering_VHH = []

    # Second round of the filter process
    VHH, target = VHH_complex_filtered
    # only does secondary check if filtering step was too stringent
    if len(target) == 0 and len(original_target) != 0:
        target_updated = []
        # checking single-chain antigens
        for c in other_chains_in_target:
            test_VHH = has_common_substring(seq_VHH, chain_dict[c]['seq'])
            if test_VHH:  # single-chain replicates
                num_contacts_greater_than_zero = check_for_stringent_condition(VHH, c, chain_dict, cutoff=12.0)
                if num_contacts_greater_than_zero >= 2:
                    chain_dict[c]["interaction_type"] = "Weak"
                    target_updated.append(c)
                    comments["Replicate_Chains"].append(c)
                    comments["Secondary_Check"].append(f'Replicate Chain {c} added to complex and has {num_contacts_greater_than_zero} by extending the cutoff distance to 12 Angstroms')
            else:  # automatically add single-chain non-replicates
                chain_dict[c]["interaction_type"] = "Strong"
                target_updated.append(c)
                comments["Secondary_Check"].append(f'Chain {c} added to complex since there were no antigens that could be added to complex')

        # testing multi-chain antigens
        for c in VH_VL_pairs_in_target:
            seq_VH_VL_pairs_in_target = [chain_dict[c[0]]['seq'], chain_dict[c[1]]['seq']]
            test_VHH = any([has_common_substring(seq_VHH, seq) for seq in seq_VH_VL_pairs_in_target])
            num_contacts_greater_than_zero_VHH = check_for_stringent_condition(VHH, c[0], chain_dict, cutoff=12.0, no_cdr1=True) + check_for_stringent_condition(VHH, c[1], chain_dict, cutoff=12.0, no_cdr1=True)
            if test_VHH:  # multi-chain antigens that are replicates
                if num_contacts_greater_than_zero_VHH >= 2:
                    chain_dict[c[0]]["interaction_type"] = "Weak"
                    chain_dict[c[1]]["interaction_type"] = "Weak"
                    target_updated += list(c)
                    comments["Replicate_Chains"].append(c[0])
                    comments["Replicate_Chains"].append(c[1])
                    comments["Secondary_Check"].append(f'Chain pair {c} added to complex and has {num_contacts_greater_than_zero_VHH} contacts by extending the cutoff distance to 12 Angstroms')
                else:
                    comments["Secondary_Check"].append(f"Chain pair {c} was not added to complex due to a lack of interactions")
            else:  # automatically add non-replicates
                if num_contacts_greater_than_zero_VHH >= 1:
                    chain_dict[c[0]]["interaction_type"] = "Strong"
                    chain_dict[c[1]]["interaction_type"] = "Strong"
                    target_updated += list(c)
                    comments["Secondary_Check"].append(f'Chain pairs {c} added to complex since there were no antigens that could be added to complex')
                else:
                    comments["Secondary_Check"].append(f"Chain pair {c} was not added to complex due to a lack of interactions")

        VHH_complex_filtered_round_2 = [list(VHH), target_updated]
    else:
        VHH_complex_filtered_round_2 = [list(VHH), list(target)]

    VHH, target = VHH_complex_filtered_round_2
    if len(target) > 0:
        # update comments with changes to complex
        if len(original_target) != len(target):
            comments["Complex_Summary"].append(f'Updated complex by removing chains {set(original_target)-set(target)}. New Complex consists of antibodies: {VHH} and antigen chains: {target}')
        else:
            comments["Complex_Summary"].append(f'Complex remains unchanged after the filter process. Complex consists of antibodies: {VHH} and antigen chains: {target}')
    else:
        # updates comments if no antigen chains are in complex
        if len(original_target) == 0:
            comments["Complex_Summary"].append(f'There were no antigen chains associated with the structure {VHH} before filter check')
        else:
            comments["Complex_Summary"].append(f'There are no antigen chains associated with the structure {VHH} after filter check possibly due to potential symmetry issue and not enough CDR contacts')

    # correcting chain_dict to only include the new chains from post filter step
    chain_dict_new = {chain: chain_dict[chain] for chain in (VHH + target)}
    return chain_dict_new, comments


def process_filter(VH, VL, Ags, chain_dict, pdb_id_old, df):
    """
    Determines if the complex involves an antibody or nanobody, and
    then determines which function should be used. Then it returns
    two dictionaries and string summarizing the complex so that it
    can be inputted into a summary csv and saved as a PDB and NPY
    file. 

    Args:
        VH (str): Chain ID of the heavy chain.
        VL (str): Chain ID of the light chain.
        Ags (list): List of the antigen chain IDs.
        chain_dict_new (dict): Contains information about the complex and
         its labeling.
        pdb_id_old (str): The pdb id of the complex before the filter function
         is applied.
        df (pd.DataFrame): DataFrame of the processed Input Structures

    Returns:
        row (dict): Dictionary containing information on the complex that
         is outputted from the autofill_row_fc function.
        chain_dict_new (dict): Dictionary containing information on the complex
         that is will be inputted into the autofill_row_fc function.
        idx (str): Details the VH, VL, and Ag invovled in the complex.
    """
    
    if VL is None:  # nanobody
        chain_dict_new, comments = filter_func_VHH(VH, Ags, chain_dict)
        idx = f"VHH_{chain_dict_new[VH]['old_id'].split('_')[0]}"
    else:  # antibody
        chain_dict_new, comments = filter_func_VH_VL([VH, VL], Ags, chain_dict)
        idx = f"VH_{chain_dict_new[VH]['old_id'].split('_')[0]}-VL_{chain_dict_new[VL]['old_id'].split('_')[0]}"

    # Adding antigen chains to idx
    Ag_chains = set(chain_dict_new[Ag]["old_id"] for Ag in chain_dict_new.keys() if Ag not in ["H", "L"])
    if len(Ag_chains) > 0:
        idx += "-Ag"
        for Ag in sorted(Ag_chains):
            idx += f"_{Ag}"

    # complex shares name with another complex
    if "replicate" in pdb_id_old:
        rep = pdb_id_old.split("replicate")[-1]
        idx += f"-replicate{rep}"

    # getting dictionary form (row) that will be added to csv
    row = autofill_row_fc(pdb_id_old, idx, chain_dict_new, comments, df)
    return row, chain_dict_new, f"{pdb_id_old.split('-VH')[0]}-{idx}"


def autofill_row_fc(pdb_id, idx, chain_dict, new_comments, df):
    """
    Returns information of the complex in a way that it can be
     stored into a summary csv that contains details on the
     complex.

    Args:
        pdb_id (str): 4 letter PDB id of structure with its
         bioassembly number.
        idx (str): Details the VH, VL, and Ag invovled in structure.
        chain_dict (dict): Contains information about the complex and
         its labeling.
        new_comments (dict): Contains information about how the complex is
         created.
        df (pd.DataFrame): Dataframe containing information about the complexes
        pd_old_id (str): PDB ID of the complex before filtering step

    Returns:
        complex_dict (dict): Dictionary containing information in structure that
         will be used to be inputted into summary csv.
    """

    parent_struct = df[df["Name"] == pdb_id]
    complex_dict = {}
    complex_dict["Name"] = f"{pdb_id.split('-VH')[0]}-{idx}" #pdb_id
    complex_dict["Parent_File"] = parent_struct["Parent_File"].values[0]
    complex_dict["Bioassembly"] = parent_struct["Bioassembly"].values[0]
    complex_dict['Structure_Title'] = parent_struct["Structure_Title"].values[0]
    complex_dict['Structure_Classification'] = parent_struct["Structure_Classification"].values[0]
    complex_dict["Resolution"] = parent_struct["Resolution"].values[0]
    complex_dict['Method'] = parent_struct["Method"].values[0]
    complex_dict['Date_Deposited'] = parent_struct["Date_Deposited"].values[0]
    complex_dict['Date_Released'] = parent_struct["Date_Released"].values[0]
    complex_dict["PDB_ID"] = parent_struct["PDB_ID"].values[0]
    complex_dict["Complex"] = idx
    complex_dict["Chain_VH"] = "H"
    complex_dict["Chain_VH_old_id"] = chain_dict["H"]["old_id"][0]
    
    if "L" in chain_dict.keys():  # antibody case
        complex_dict["Chain_VL"] = "L"
        complex_dict["Chain_VL_old_id"] = chain_dict["L"]["old_id"][0]
    else:  # nanobody case
        complex_dict["Chain_VL"] = None
        complex_dict["Chain_VL_old_id"] = None

    complex_dict["Chain_Ag"] = sorted(set(chain_dict.keys()) - set(["H", "L"]))
    complex_dict["Chain_Ag_old_id"] = [chain_dict[Ag]['old_id'] for Ag in complex_dict["Chain_Ag"]]
    complex_dict["Interaction_Type"] = [chain_dict[Ag]['interaction_type'] for Ag in complex_dict["Chain_Ag"]]
    complex_dict["Antigen_Type"] = [chain_dict[Ag]['antigen_type'] for Ag in complex_dict["Chain_Ag"]]
    prev_comments = ast.literal_eval(parent_struct["comments"].values[0])
    comments = {key: prev_comments[key] + new_comments[key] for key in prev_comments}
    complex_dict["comments"] = comments
    return complex_dict


def add_pdb_header_fc(pdb_file, input_dir, pdb_id, chain_dict):
    """
    Adds all the header information from the original PDB file into the new one.

    Args:
        pdb_file (str): Newly created PDB file
        input_dir (str): Path to input directory.
        pdb_id (dict): File name of the original PDB structure
        chain_dict (dict): Dictionary containing information on the structure
    """
    pdb_name = pdb_file.split("/")[-1][:-4]
    # Reading the original file content
    with open(pdb_file, "r") as f_read:
        content = "".join(f_read.readlines()[1:])

    with open(pdb_file, "w") as f_new:
        # Adding header information
        with open(f"{input_dir}/{pdb_id}.pdb", 'r') as f_old:
            for line in f_old:
                if len(line) > 8:
                    if line.split()[0] == "STRUCTURE_IDENTITY":
                        complex_name = pdb_name.split("-")[1:]
                        remark = []
                        if "VHH" in pdb_name:
                            VHH = pdb_name.split('-VHH_')[-1][0]
                            remark.append(f"The VHH Chain ID is H, with its old ID being {VHH}")
                        else:
                            VH = pdb_name.split('-VH_')[-1][0]
                            VL = pdb_name.split('-VL_')[-1][0]
                            remark.append(f"The VH Chain ID is H, with its old ID being {VH}")
                            remark.append(f"The VL Chain ID is L, with its old ID being {VL}")

                        Ag_chain = sorted(set(chain_dict.keys()) - set(["H", "L"]))
                        if Ag_chain:
                            Ag = ", ".join(Ag_chain)
                            Ag_old_id = ", ".join([chain_dict[Ag_id]["old_id"] for Ag_id in Ag_chain])
                            remark.append(f"The Antigen Chain ID is {Ag}, with its old ID being {Ag_old_id}")
                        remark = "; ".join(remark)
                        f_new.write(f"STRUCTURE_IDENTITY  {remark}\n")
                        f_new.write("REMARK 99 Modified by Bryan Munoz Rivero, Sanofi\n")
                        break
                    else:
                        f_new.write(line)
        f_new.write(content)  # Adding the original content


def pipeline_func(pdb_id, df, complex_dir, filter_dir, pdb_utils):
    """
    Filters out chains not involved in the complex. Returns this 
     information as a list of dictionaries, where each
     dictionary contains information about a filtered complex.

    Args:
        pdb_id (str): name of complex.
        df (pd.DataFrame): Dataframe containing information on the
         compleses.
        complex_dir (str): path to directory where the complexes are
         inputted from
        filter_dir (str): path to directory where the complexes will 
         be saved
        pdb_utils (PDBUtils): An instance of the PDBUtils class that is used
         to process and label the chains.

    Returns:
        row (dict): Dictionary that contains information about the post-filtered
         complex.
    """

    # call information about the complex
    pdb = df.loc[df["Name"] == pdb_id]

    # pulling up the chain ID for VH, VL
    Chain_VH = "H"
    Chain_VL = "L" if "-VHH_" not in pdb_id else None

    # determining the antigen chains of the complex
    npy_file = f'{complex_dir}/{pdb_id}-atom37.npy'
    chain_dict = np.load(npy_file, allow_pickle=True).item()
    ag_chains = str(pdb['Chain_Ag'].values[0])
    Ags = ast.literal_eval(ag_chains)

    # filtering function for complex and creating new structure files for complex
    row, chain_dict_new, pdb_id_new = process_filter(Chain_VH, Chain_VL, Ags, chain_dict, pdb_id, df)
    structure = dict_to_structure(chain_dict_new)
    output_file = os.path.join(filter_dir, pdb_id_new)
    pdb_utils.write_pdb_with_seqres_from_structure(structure[0], output_file + ".pdb")
    add_pdb_header_fc(output_file + ".pdb", complex_dir, pdb_id, chain_dict_new)
    np.save(output_file + "-atom37.npy", chain_dict_new)
    print("Successfully Completed:", output_file)
    return row


def main():
    """
    The main function of the third part of the pipeline. The purpose of this script
     is to look at each complex and determine which antigen chains are truly invovled
     in the complex to see if they should be filtered out.
    """

    # define inputs
    parser = argparse.ArgumentParser(description="Filter out chains that are do not have interactions with the binder.")
    # Add arguments
    parser.add_argument(
        "input_dir",
        help="Path to the input PDB or CIF directory. If omitted, the --test option must be used.")
    args = parser.parse_args()
    pdb_utils = PDBUtils()
    pdb_folder = args.input_dir
    if pdb_folder[-1] == "/":
        pdb_folder = pdb_folder[:-1]
    pdb_folder = pdb_folder if "/" in pdb_folder else "./" + pdb_folder

    #deleting pre-existing directories
    if os.path.exists(f"{pdb_folder}_filter"):
        shutil.rmtree(f"{pdb_folder}_filter")
    if os.path.exists(f"{pdb_folder}_outputs_multichain_filter.csv"):
        os.remove(f"{pdb_folder}_outputs_multichain_filter.csv")

    # define input and output folders, and output csv, and reads previous summary csv
    output_csv_name = f'{pdb_folder}_outputs_multichain_filter.csv'
    complex_dir = f'{pdb_folder}_complexes'
    filter_dir = f"{pdb_folder}_filter"
    os.makedirs(filter_dir, exist_ok=True)
    df = pd.read_csv(f'{pdb_folder}_complexes_curated.csv')

    # running process_pdb through parallel function
    pdb_names = sorted(df["Name"].values)
    cpu_parallel = ParallelProcessorForCPUBoundTasks(pipeline_func, max_workers=48)
    processed_rows = cpu_parallel.process(pdb_names, df, complex_dir, filter_dir, pdb_utils)
    
    # saving the recorded information for each structure into the csv
    processed_df = pd.DataFrame(processed_rows)
    processed_df = processed_df.sort_values(by="Name")
    processed_df.to_csv(output_csv_name, index=False)
    processed_df.info()


if __name__ == "__main__":
    main()