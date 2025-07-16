import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
from anarci import run_anarci
import re
from Bio.PDB.Polypeptide import protein_letters_3to1
import warnings


def get_region_splits_anarci(seq):
    """
    The purpose of this function is to leverage ANARCI to get relevant
    and needed information about the identity of a sequence. This
    includes the type of the chain (Heavy, Light, a TCR chain, etc.),
    IMGT numbering of the chain, and the partitioning of the chain. 

    Args:
        seq (str): Inputted Amino Acid Sequence

    Returns:
        region_splits (dict): Describing the sequence for the partitions
        index_imgt (list): IMGT Numbering of chain
        anarci_out[2][0][0]['chain_type'] (str): Identity of chain type
    """
    
    region_splits = {
        f"fwr1": '' ,
        f"cdr1":  '' ,
        f"fwr2":  '',
        f"cdr2":  '',
        f"fwr3":  '',
        f"cdr3": '' ,
        f"fwr4": '' ,
     }
    ranges = [(1, 26), (27, 38), (39, 55), (56, 65), (66, 104), (105, 117), (118, 128)]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Limiting hmmer search to species")
        anarci_out = run_anarci(seq, scheme="imgt")
    anarci_numbering = anarci_out[1][0][0][0]
    index = []
    index_imgt = []
    seq_reconst = ""
    # ignore "-"
    for element in anarci_numbering:
        idx = str(element[0][0])
        if element[0][1] != " ":
            idx += element[0][1]  # (1, 'A') -> '1A'
        aa = element[1]  # 'Q' or '-'
        index.append(idx)
        seq_reconst += aa
        
        if aa != "-":
            index_imgt.append(idx)

    for name, rng in zip(list(region_splits.keys()), ranges):
        st = rng[0]
        e = rng[1]

        try:
            idx_st = index.index(str(st))
        except:
            
            assert name=='fwr4', "Only framework 4 can be missing"

            continue
            
        try:
            idx_e = index.index(str(e + 1)) 
        except:
            idx_e = None
            assert name=='fwr4' or name=='cdr3', "Only framework 4 can be partially missing"

        region_splits[name] = seq_reconst[idx_st:idx_e].replace("-", "")

    # add constant region
    const_parts = seq.split(seq_reconst.replace("-", ""))
    index_imgt = (
        ["C" for i in range(len(const_parts[0]))]
        + index_imgt
        + ["C" for i in range(len(const_parts[1]))]
    )
    
    region_splits.update({'c_st': const_parts[0], 'c_e': const_parts[1]})
    
    return region_splits, index_imgt, anarci_out[2][0][0]['chain_type']


def eliminate_non_aa(input_string):
    """
    Purpose of this function is to change nonstandard AA to X.

    Parameters:
    - input_string (str): string of the input sequence.

    Returns:
    - result (str): a string with nonstandard AA now having the ID X
    """
    
    # Define the pattern to match substrings of the type "(*)"
    pattern = r'\([^)]*\)'
    
    # Use re.sub() to replace all occurrences of the pattern with an empty string
    result = re.sub(pattern, '', input_string)

    result = ''.join([c if c in protein_letters_3to1.values() else 'X' for c in result])
    
    return result


def get_anarci_pos_synthetic_contructs(seq):
    """
    Performs anarci on a provided sequence to determine if it is heavy or light chain
    
    Parameters:
    - seq (str): Sequence of the chain.

    Returns:
    - all_chain_type (list): Identifier of what the chain is
    - all_index (list): List of IMGT numbering of sequences
    - all_seq_reconst (list): List of the sequence broken into segments
        by anarci
    """
    
    anarci_out = run_anarci(seq, scheme="imgt")
    all_index = []
    all_chain_type = []
    all_seq_reconst = []
    for i in range(len(anarci_out[1][0])):
        anarci_numbering = anarci_out[1][0][i][0]
        index = []
        seq_reconst = ""
        # ignore "-"
        for element in anarci_numbering:
            idx = str(element[0][0])
            if element[0][1] != " ":
                idx += element[0][1]  # (1, 'A') -> '1A'
            aa = element[1]  # 'Q' or '-'
            if aa != "-":
                index.append(idx)
                seq_reconst += aa

        all_index.append(index)
        all_seq_reconst.append(seq_reconst)
        all_chain_type.append(anarci_out[2][0][i]['chain_type'])
        
    return all_chain_type, all_index, all_seq_reconst


def chain_dict_to_VH_VL_Ag_categories(
    input_chain_dict,
    override_assignments=False,
    default_assignments={'VH': 'H', 'VL': 'L'}
):
    """Parse input chains into VH, VL, and Ag categories.

    ANARCI returns chain_type ∈ {'H','L','K','A','B','G','D'}:
      H = Ab heavy, L = Ab lambda, K = Ab kappa,
      A = TCR α, B = TCR β, G = TCR γ, D = TCR δ.
    """
    HEAVY = {'H','B','D'}
    LIGHT = {'L','K','A','G'}
    chain_dict = {'VH': {}, 'VL': {}, 'Ag': {}}
    parsed = {}

    for cid, data in input_chain_dict.items():
        parsed[cid] = data.copy()
        try:
            types, idxs, seqs = get_anarci_pos_synthetic_contructs(data['seq'])
            for i, ctype in enumerate(types):
                key = f"{cid}_{i}"
                match = next(re.finditer(seqs[i], data['seq']))
                start, end = match.span()
                fragment = {
                    'chain_type': ctype,
                    'imgt': idxs[i],
                    'seq': seqs[i],
                    'atom37': data['atom37'][start:end],
                }
                if 'b_factor' in data:
                    fragment['b_factor'] = data['b_factor'][start:end]
                parsed[key] = fragment
        except Exception as e:
            continue

    for key, val in parsed.items():
        ctype = val.get('chain_type')
        if override_assignments:
            if default_assignments['VH'] + '_' in key:
                chain_dict['VH'][key] = val
            elif default_assignments['VL'] + '_' in key:
                chain_dict['VL'][key] = val
            else:
                chain_dict['Ag'][key] = val
        else:
            if ctype in HEAVY:
                chain_dict['VH'][key] = val
            elif ctype in LIGHT:
                chain_dict['VL'][key] = val
            else:
                chain_dict['Ag'][key] = val

    return chain_dict