import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import argparse

# the dataset can be downloaded from
# https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help
# nmrshiftdb2withsignals.sd


molsuppl = Chem.SDMolSupplier('./data/nmrshiftdb2withsignals.sd', removeHs = False)


atom_list = ['Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
# atom_list=['C','N','O','F','P','S','Cl']
charge_list = [1, 2, 3, -1, -2, -3, 0]
degree_list = [1, 2, 3, 4, 5, 6, 0]
valence_list = [1, 2, 3, 4, 5, 6, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]
n_max = 64
dim_edge = 10

bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
stereo_list = ['STEREOZ', 'STEREOE','STEREOANY','STEREONONE']

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

def get_atom_shifts_13C(mol):
    
    molprops = mol.GetPropsAsDict()
    
    atom_shifts = {}
    for key in molprops.keys():
        if key.startswith('Spectrum 13C'):
            for shift in molprops[key].split('|')[:-1]:
            
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
            
                if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []
                atom_shifts[shift_idx].append(shift_val)

    return atom_shifts

def get_atom_shifts_1H(mol):
    
    molprops = mol.GetPropsAsDict()
    
    atom_shifts = {}
    for key in molprops.keys():
        if key.startswith('Spectrum 1H'):
            tmp_dict = {}
            for shift in molprops[key].split('|')[:-1]:
            
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
            
                if shift_idx not in tmp_dict: tmp_dict[shift_idx] = []
                tmp_dict[shift_idx].append(shift_val)
                

            for shift_idx in tmp_dict.keys():
                tmp_dict[shift_idx] = np.mean(tmp_dict[shift_idx])
                
                if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []
                atom_shifts[shift_idx].append(tmp_dict[shift_idx])

    return atom_shifts

def _DA(mol):

    D_list, A_list = [], []
    for feat in chem_feature_factory.GetFeaturesForMol(mol):
        if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
        if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
    
    return D_list, A_list

def _chirality(atom):

    if atom.HasProp('Chirality'):
        #assert atom.GetProp('Chirality') in ['Tet_CW', 'Tet_CCW']
        c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
    else:
        c_list = [0, 0]

    return c_list
    

def _stereochemistry(bond):

    if bond.HasProp('Stereochemistry'):
        #assert bond.GetProp('Stereochemistry') in ['Bond_Cis', 'Bond_Trans']
        s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
    else:
        s_list = [0, 0]

    return s_list    

def bondFeatures(bid1, bid2, mol, rings):
    atom_idx = [a.GetIdx() for a in mol.GetAtoms()]
    
    bondpath = Chem.GetShortestPath(mol, bid1, bid2)
    bonds = [mol.GetBondBetweenAtoms(bondpath[t], bondpath[t + 1]) for t in range(len(bondpath) - 1)]

    samering = 0
    for ring in rings:
        if bid1 in ring and bid2 in ring:
            samering = 1

    if len(bonds)==1:
        b = mol.GetBondBetweenAtoms(atom_idx[bid1], atom_idx[bid2])
        bond_fea1 = np.eye(len(bond_list), dtype = bool)[bond_list.index(str(b.GetBondType()))]
        bond_fea2 = np.array(_stereochemistry(b), dtype = bool)
        bond_fea3 = np.array([b.IsInRing(), b.GetIsConjugated()])
    else:
        bond_fea1 = np.zeros(4)
        bond_fea2 = np.zeros(2)
        bond_fea3 = np.zeros(2)

    bond_fea4 = np.array([len(bonds), samering])
        
    return np.concatenate([bond_fea1, bond_fea2, bond_fea3, bond_fea4], axis=0)

def add_mol(mol_dict, mol, nmr_mode):

    n_node = mol.GetNumAtoms()
    n_edge = n_node * (n_node -1)

    D_list, A_list = _DA(mol)
    rings = mol.GetRingInfo().AtomRings()
    
    if nmr_mode == '13C':
    
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    elif nmr_mode == '1H':
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
        #atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs()) for a in mol.GetAtoms()]][:,:-1]

        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        
        #Using mask prop: binary feature for presence of hydrogen
        atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing(), a.GetBoolProp('mask')] for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    
    if n_edge > 0:

        # edge_attr = np.zeros((n_node, n_node, dim_edge), dtype=np.int8)
        edge_attr = []
        bond_loc = [[],[]]
        for j in range(n_node):
            for k in range(j+1, n_node):
                edge_attr.append(bondFeatures(j, k, mol, rings))
                bond_loc[0].append(j)
                bond_loc[1].append(k)
        edge_attr = np.vstack([edge_attr, edge_attr])
        bond_loc = np.array(bond_loc)
        src = np.hstack([bond_loc[0,:], bond_loc[1,:]])
        dst = np.hstack([bond_loc[1,:], bond_loc[0,:]])
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)
    
    return mol_dict

def preprocess(args):
    target = args.target
    length = len(molsuppl)
    mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'shift': [],
                'mask': [],
                'smi': []}
    k_list=[]
    k=0                
    a, b, c, d, e = 0, 0, 0, 0, 0
    for i, mol in enumerate(molsuppl):
        # if i == 1000:
        #     break
        #print(i)
        try:   
            Chem.SanitizeMol(mol)
            Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
            Chem.rdmolops.AssignStereochemistry(mol)
            assert '.' not in Chem.MolToSmiles(mol)
        except:
            a += 1
            continue

        if target == '13C': 
            atom_shifts = get_atom_shifts_13C(mol)
        elif target == '1H':
            atom_shifts = get_atom_shifts_1H(mol)

        if len(atom_shifts) == 0: 
            b += 1
            continue

        for j, atom in enumerate(mol.GetAtoms()):
            if j in atom_shifts:
                atom.SetDoubleProp('shift', np.median(atom_shifts[j]))
                atom.SetBoolProp('mask', 1)
            else:
                atom.SetDoubleProp('shift', 0)
                atom.SetBoolProp('mask', 0)

        mol = Chem.RemoveHs(mol)

        # if mol.GetNumAtoms() > n_max:
        #     c += 1
        #     continue

        # atom_set = set([a.GetSymbol() for a in mol.GetAtoms()])
        # if len(atom_set - set(atom_list)) > 0:
        #     d += 1
        #     continue

        if 'H' in [at.GetSymbol() for at in mol.GetAtoms()]:
            e += 1
            continue

        mol_dict = add_mol(mol_dict, mol, target)

        
        if (i+1)%1000==0: 
            print(f'{i+1}/{length} processed')

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    mol_dict['shift'] = np.hstack(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])
    print(a,b,c,d,e, a+b+c+d+e)
    for key in mol_dict.keys(): print(key, mol_dict[key].shape, mol_dict[key].dtype)
        
    np.savez_compressed('./data/nmrshiftdb2_graph_%s_full.npz'%target, data = [mol_dict])


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--target', help ='13C or 1H', default='1H', type = str)
    args = arg_parser.parse_args()

    preprocess(args)