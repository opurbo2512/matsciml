from material import Material
import numpy as np
import pandas as pd
import torch
import time

#Family A — Lattice & Geometry Features (8 features)

class LatticeFeaturizer:
    def __init__(self):
        self._column_names = ["a","b","c","alpha","beta","gamma","volume","packing_fraction"]

    def transform(self,materials):
        lattice_info = []
        for material in materials:
            """
            Removing the attribute and making list directly.
            """
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
            lattice_info.append(lattice_list)

        lattice_array = np.array(lattice_info)
        return lattice_array
    

#Family B — Bond & Environment Features (8 features)

class BondFeaturizer:
    def __init__(self):
        info = []
        self._column_names = [
            "bond length","minumum bond length","maximum bond length",
                "bond length std","coordination number","bond angle mean",
                "bond angle std","econ"
        ]

    def transform(self,materials):
        info = []
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
                bond_length_std,coordination_number,bond_angle_mean,
                bond_angle_std,econ
            ]
            info.append(info_list)

        bond_array = np.array(info)
        return bond_array
    

#Family C — Compositional & Electronic Features

class CompositionFeaturizer:
    def __init__(self):
        self._column_names = [
                "electronegativity difference","electronegativity mean","valence electron per atom","d electron per atom",
                "avg ionization energy","avg electron affinity","cation to anion ratio",
                "ionic character","avg atomic mass","atomic number variance"
            ]

    def transform(self,materials):
        comp_info = []

        for material in materials:
            electronegativity_difference = material.electronegativity_diff
            electronegativity_mean = material.electronegativity_mean
            valence_electron_per_atom = material.valence_electron_count
            d_electron_per_atom = material.d_electron_per_atom
            avg_ionization_energy = material.avg_ionization_energy
            avg_electron_affinity = material.avg_electron_affinity
            cation_to_anion_ratio = material.cation_to_anion_ratio
            ionic_character = material.ionic_character
            avg_atomic_mass = material.avg_atomic_mass
            atomic_number_variance = material.atomic_number_variance

            info_list = [
                electronegativity_difference,electronegativity_mean,valence_electron_per_atom,d_electron_per_atom,
                avg_ionization_energy,avg_electron_affinity,cation_to_anion_ratio,
                ionic_character,avg_atomic_mass,atomic_number_variance
            ]
            comp_info.append(info_list)

        comp_info = np.array(comp_info)
        return comp_info


#Family D — Symmetry & Topology Features
class SymmetryFeaturizer:
    def __init__(self):
        self.symmetry_list = ["cubic","hexagonal","trigonal","tetragonal","orthorhombic","monoclinic"]
        self._column_names = self.symmetry_list
        
    def transform(self,materials):
        symmetry_info = []

        for material in materials:
            zero_array = np.zeros(6)
            crystal_type = str(material.crystal_system).lower().strip()

            try:
                index_cubic = self.symmetry_list.index(crystal_type)
                zero_array[index_cubic] = 1
            except:
                pass

            symmetry_info.append(zero_array)
        
        return np.array(symmetry_info)
    

#main class
class StructuralFeaturizer:

    def __init__(self,include="all",exclude=None,
                 normalize='standard', output_format='numpy'):
        
        self._lattice= LatticeFeaturizer()
        self._bond= BondFeaturizer()
        self._comp= CompositionFeaturizer()
        self._symmetry= SymmetryFeaturizer()
        
        all_features = ["lattice","bond","comp","symmetry"]

        if include == "all":
            self._info_needed_list = all_features.copy()

        else:
            self._info_needed_list = list(include)

        if exclude is not None:
            self._info_needed_list = ["lattice","bond","comp","symmetry"]
            for info in exclude:
                if info in self._info_needed_list:
                    self._info_needed_list.remove(info)

        self.column_names = []
        if "lattice" in self._info_needed_list:
            self.column_names += self._lattice._column_names
        if "bond" in self._info_needed_list:
            self.column_names += self._bond._column_names
        if "comp" in self._info_needed_list:
            self.column_names += self._comp._column_names
        if "symmetry" in self._info_needed_list:
            self.column_names += self._symmetry._column_names

        self._output_format = output_format
        self._fitted = False
        self._normalize = normalize
            
    
    def _extract_features(self,materials):

        feature_map = {
            "lattice": self._lattice,
            "bond": self._bond,
            "comp": self._comp,
            "symmetry": self._symmetry,
        }

        arrays = [
            feature_map[name].transform(materials)
            for name in self._info_needed_list
        ]

        return np.hstack(arrays)
    
    def fit(self,materials):
        X = self._extract_features(materials)
        
        if "symmetry" in self._info_needed_list:
            X_cont = X[:,:-6]
            X_cat = X[:,-6:]
        else:
            X_cont = X
        if self._normalize == "standard":
            self.std_ = np.std(X_cont,axis=0)
            self.std_ = np.where(self.std_ == 0, 1, self.std_)
            self.mean_ = np.mean(X_cont,axis=0)
            self._scalar_up = self.mean_
            self._scalar_down = self.std_

        elif self._normalize == "minimax":
            self.min_ = np.min(X_cont,axis=0)
            self.max_ = np.max(X_cont,axis=0)
            self._scalar_up = self.min_
            self._scalar_down = self.max_ - self.min_
            self._scalar_down = np.where(self._scalar_down == 0, 1, self._scalar_down)

        elif self._normalize == "robust":
            self.q25_ = np.percentile(X_cont,25,axis=0)
            self.q75_ = np.percentile(X_cont,75,axis=0)
            self.median_ = np.median(X_cont,axis=0)
            self._scalar_up = self.median_
            self._scalar_down = self.q75_ - self.q25_
            self._scalar_down = np.where(self._scalar_down == 0, 1, self._scalar_down)

        else:
            self._scalar_up = 0
            self._scalar_down = 1
        if "symmetry" in self._info_needed_list:
            X = np.hstack((X_cont,X_cat))
        else:
            X = X_cont

        self._fitted = True
        return self

    def transform(self,materials):
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        X = self._extract_features(materials)

        if "symmetry" in self._info_needed_list:
            X_cont = X[:, :-6]
            X_cat = X[:, -6:]
        else:
            X_cont = X

        X_cont = (X_cont - self._scalar_up) / self._scalar_down

        if "symmetry" in self._info_needed_list:
            X = np.hstack((X_cont, X_cat))
        else:
            X = X_cont

        return X
    
    def fit_transform(self,materials):
        return self.fit(materials).transform(materials)
    
    def _format_output(self,X,materials):
        if self._output_format == "numpy":
            return X
        elif self._output_format == "dataframe":
            df = pd.DataFrame(X,columns=self.column_names)
            return df
        elif self._output_format == "tensor":
            tensor = torch.from_numpy(X)
            return tensor
        elif self._output_format == "dict":
            df = pd.DataFrame(X,columns=self.column_names)
            return df.to_dict(orient="list")
            
