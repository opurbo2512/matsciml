from material import Material
import numpy as np
import time

#Family A — Lattice & Geometry Features (8 features)

class LatticeFeaturizer:
    def __init__(self):
        self._lattice_info = []

    def tansform(self,materials):
        for material in materials:
            a = material.a
            b = material.b
            c = material.c
            alpha = material.alpha
            beta = material.beta
            gamma = material.gamma
            volume = material.cell_volume
            packing_frac = material.packing_fraction

            lattice_list = [
                a,b,c,
                alpha,beta,gamma,
                volume,packing_frac
            ]
            self._lattice_info.append(lattice_list)

        lattice_array = np.array(self._lattice_info)
        return lattice_array
    

#Family B — Bond & Environment Features (8 features)

class BondFeaturizer:
    def __init__(self):
        self._info = []

    def transform(self,materials):
        for material in materials:
            bond_length = material.bond_length
            bond_length_min = material.bond_length_min
            bond_length_max = material.bond_length_max
            bond_length_std = material.bond_length_std
            coordination_number = material.coordination_number
            bond_angle_mean = material.bond_angle_mean
            bond_angle_std = material.bond_angle_std
            econ = material.econ

            info_list = [
                bond_length,bond_length_min,bond_length_max,
                bond_angle_std,coordination_number,bond_angle_mean,
                bond_angle_std,econ
            ]
            self._info.append(info_list)

        bond_array = np.array(self._info)
        return bond_array
    

#Family C — Compositional & Electronic Features

class CompositionFeaturizer:
    pass


#Family D — Symmetry & Topology Features
class SymmetryFeaturizer:
    def __init__(self):
        self._info = []
        self.symmetry_list = ["cubic","hexagonal","trigonal","tetragonal","orthorhombic","monoclinic","triclinic"]
        
    def transform(self,materials):
        for material in materials:
            zero_array = np.zeros(7)
            crystal_type = material.crystal_system
            index_cubic = self.symmetry_list.index(crystal_type)
            zero_array[index_cubic] = 1
            self._info.append(zero_array)
        
        return self._info
    
        
    



