#standard python libraries
import os
import time
import math
import logging
import warnings
from itertools import combinations

#scientifically computing and machine learning
import numpy as np
import torch

#material science library (Pymatgen)
from pymatgen.core import Composition
from pymatgen.core import Element
from pymatgen.core import Structure
from pymatgen.core import Lattice

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.local_env import EconNN
from pymatgen.analysis.bond_valence import BVAnalyzer

#material project api
from mp_api.client import MPRester

#graph neaural network model
import matgl

#project specific module
from composition import CompositionEngine

#environment configure
os.environ["HF_TOKEN"] = "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"
warnings.filterwarnings("ignore")
logging.getLogger("pymatgen").setLevel(logging.ERROR)


class Material:
    def __init__(self,input_formula):
        self.input_formula = input_formula
        self._structure = None
        self._sga = None

        #general attributes
        self._formula = None
        self._name = None #do this later
        self._elements = None
        self._n_elements = None
        self._source = None

        #composition attributes
        self._composition = None
        self._electronegativity_diff = None
        self._oxidation_states = None # there are some bug, should update later
        self._valence_electrons = None
        self._ionic_character = None
        self._crystal_type = None # do this later
        self._avg_atomic_mass = None

        #structure attributes
        self._crystal_system = None
        self._space_group = None
        self._space_group_number = None
        self._lattice = None
        self._a = None
        self._b = None
        self._c = None
        self._alpha = None
        self._beta = None
        self._gamma = None
        self._cell_volume = None
        self._density = None
        self._n_atoms_unit_cell = None
        self._n_symmetry_ops = None
        self._is_centrosymmetric = None

        #environment attributes
        self._is_polar = None
        self._coordination_number = None
        self._bond_length = None
        self._bond_length_min = None 
        self._bond_length_max = None 
        self._bond_length_std = None
        self._packing_fraction = None
        self._bond_angle_mean = None
        self._bond_angle_std = None
        self._econ = None #(Effective Coordination Number)
        
        #electronic attributes
        self._valence_electron_count = None
        self._d_electron_count = None
        self._bandgap_estimate = None
        self._bandgap_ml = None
        self._electronic_type = None
        self._avg_ionization_energy = None

        if self.input_formula is not None:
            self._run_initial()

    def _run_initial(self):
        self.input_formula = self.input_formula.replace(" ","")
        API_KEY = "ihddElrSgFw4L4DlUUb4VkH1px5SYhS0"
        with MPRester(API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula = self.input_formula
            )
        if len(docs) == 0:
            raise ValueError(
                f"No material found for {self.input_formula}"
            )
        best = min(
            docs,
            key = lambda x : x.energy_above_hull
        )
        self._structure = best.structure
        self._sga = SpacegroupAnalyzer(self._structure)
        self._source = "Chemical Formula"

    @classmethod
    def from_cif(cls,file_name):
        instance = cls(input_formula = None)
        instance._structure = Structure.from_file(file_name)
        instance._sga = SpacegroupAnalyzer(instance._structure)
        instance._source = "Cif file"

        return instance
    
    @classmethod
    def from_poscar(cls,file_name):
        instance = cls(input_formula = None)
        instance._structure = Structure.from_file(file_name)
        instance._sga = SpacegroupAnalyzer(instance._structure)
        instance._source = "Poscar file"

        return instance

    
    @classmethod
    def from_mp(cls,mp_id):
        instance = cls(input_formula = None)
        API_KEY = "ihddElrSgFw4L4DlUUb4VkH1px5SYhS0"
        with MPRester(API_KEY) as mpr:
            instance._structure = mpr.get_structure_by_material_id(mp_id)
        instance._sga = SpacegroupAnalyzer(instance._structure)
        instance._source = "MP(Material Project) id"

        return instance
    
    @classmethod
    def from_dict(cls,dic):
        instance = cls(input_formula=None)
        lattice_dic = dic["lattice"]
        lattice = Lattice.from_parameters(
                    lattice_dic["a"],lattice_dic["b"],lattice_dic["c"],
                    lattice_dic["alpha"],lattice_dic["beta"],lattice_dic["gamma"]
                )
        species = dic["species"]
        coords = dic["coords"]
        instance._structure = Structure(lattice,species,coords)
        instance._sga = SpacegroupAnalyzer(instance._structure)
        instance._source = "Dictionary of lattice,species and coordination."

        return instance

    @property
    def formula(self):
        if self._formula is None:
            self._formula = self._structure.composition.reduced_formula
        return self._formula
    
    """
    @property
    def name(self):
        if self._name is None:
            pass
        return self._name
    """

    @property
    def elements(self):
        if self._elements is None:
            self._elements = []
            for element in self._structure.elements:
                ele_name = element.symbol
                self._elements.append(ele_name)
        return self._elements
    
    @property
    def n_elements(self):
        if self._n_elements is None:
            list_of_elements = self._structure.elements
            self._n_elements = len(list_of_elements)
        return self._n_elements
    
    @property
    def source(self):
        return self._source
    
    @property
    def composition(self):
        if self._composition is None:
            chem_formula = self._structure.formula
            self._composition = CompositionEngine(chem_formula)
        return self._composition
    
    @property
    def electronegativity_diff(self):
        if self._electronegativity_diff is None:
            list_of_elements = self._structure.elements
            list_of_electo = []
            for element in list_of_elements:
                list_of_electo.append(element.X)
            self._electronegativity_diff = max(list_of_electo) - min(list_of_electo)
        return self._electronegativity_diff
    
    @property
    def oxidation_states(self):
        if self._oxidation_states is None:
            self._oxidation_states = {}
            try:
                bva = BVAnalyzer()
                structure_oxi = bva.get_oxi_state_decorated_structure(self._structure)

                for site in structure_oxi:
                    specie = site.specie
                    symbol = specie.symbol
                    state = specie.oxi_state
                    self._oxidation_states[symbol] = state
            except:
                ele = self._structure.elements[0].symbol
                self._oxidation_states[ele] = 0

        return self._oxidation_states
    
    @property
    def valence_electrons(self):
        if self._valence_electrons is None:
            list_of_elements = self._structure.elements
            self._valence_electrons = {}
            for element in list_of_elements:
                symbol = element.symbol
                highest_n = max(n for n,_,_ in element.full_electronic_structure)
                valence_electrons = sum(
                    occ
                    for n,_,occ in element.full_electronic_structure
                    if n == highest_n
                )
                self._valence_electrons[symbol] = valence_electrons

        return self._valence_electrons
    
    @property
    def ionic_character(self):
        if self._ionic_character is None:
            en_diff = self.electronegativity_diff
            self._ionic_character = (1-np.exp(-((en_diff ** 2)/4))) * 100

        return round(self._ionic_character,4)
    
    """
    @property
    def crystal_type(self):
        if self._crystal_type is None:
            pass

        return self._crystal_type
    """
    
    @property
    def avg_atomic_mass(self):
        if self._avg_atomic_mass  is None:
            self._avg_atomic_mass = self._structure.composition.weight
        return round(self._avg_atomic_mass ,4)
    
    @property
    def structure(self):
        return self._structure
    
    @property
    def crystal_system(self):
        if self._crystal_system is None:
            self._crystal_system = self._sga.get_crystal_system()

        return self._crystal_system
    
    @property
    def space_group(self):
        if self._space_group is None:
            self._space_group = self._sga.get_space_group_symbol()

        return self._space_group
    
    @property
    def space_group_number(self):
        if self._space_group_number is None:
            self._space_group_number = self._sga.get_space_group_number()

        return self._space_group_number
    
    @property
    def lattice(self):
        if self._lattice is None:
            latt = self._structure.lattice
            self._lattice = {
                "a" : latt.a,
                "b" : latt.b,
                "c" : latt.c,
                "alpha" : latt.alpha,
                "beta" : latt.beta,
                "gamma" : latt.gamma
            }
        return self._lattice
    
    @property
    def a(self):
        if self._a is None:
            lat_dic = self.lattice
            self._a = lat_dic["a"]
        return self._a
    
    @property
    def b(self):
        if self._b is None:
            lat_dic = self.lattice
            self._b = lat_dic["b"]
        return self._b
    
    @property
    def c(self):
        if self._c is None:
            lat_dic = self.lattice
            self._c = lat_dic["c"]
        return self._c
    
    @property
    def alpha(self):
        if self._alpha is None:
            lat_dic = self.lattice
            self._alpha = lat_dic["alpha"]
        return round(self._alpha,4)
    
    @property
    def beta(self):
        if self._beta is None:
            lat_dic = self.lattice
            self._beta = lat_dic["beta"]
        return round(self._beta,4)
    
    @property
    def gamma(self):
        if self._gamma is None:
            lat_dic = self.lattice
            self._gamma = lat_dic["gamma"]
        return round(self._gamma,4)
    
    @property
    def cell_volume(self):
        if self._cell_volume is None:
            self._cell_volume = self._structure.volume
        return round(self._cell_volume,4)

    @property
    def density(self):
        if self._density is None:
            self._density = self._structure.density
        return round(self._density,4)
    
    @property
    def n_atoms_unit_cell(self):
        if self._n_atoms_unit_cell is None:
            self._n_atoms_unit_cell = len(self._structure)
        return self._n_atoms_unit_cell
    
    @property
    def n_symmetry_ops(self):
        if self._n_symmetry_ops is None:
            sym_ops = self._sga.get_symmetry_dataset()
            self._n_symmetry_ops = len(sym_ops)
        return self._n_symmetry_ops
    
    @property
    def is_centrosymmetric(self):
        if self._is_centrosymmetric is None:
            symm_ops = self._sga.get_symmetry_operations()
            self._is_centrosymmetric = False
            for op in symm_ops:
                if op.rotation_matrix.trace() == -3:
                    self._is_centrosymmetric = True
                    break
        return self._is_centrosymmetric
    
    @property
    def is_polar(self):
        if self._is_polar is None:
            polar_point_group = {
                "1","2","m","mm2",
                "3","3m",
                "4","4mm",
                "6","6mm"
            }
            point_group_sym = self._sga.get_point_group_symbol()
            self._is_polar = False
            if point_group_sym in polar_point_group:
                self._is_polar = True
        return self._is_polar
    
    @property
    def coordination_number(self):
        if self._coordination_number is None:
            cnn = CrystalNN()
            coordination_numbers = []

            for i in range(len(self._structure)):
                cn = cnn.get_cn(self._structure , i)
                coordination_numbers.append(cn)
            
            self._coordination_number = np.mean(coordination_numbers)

        return round(self._coordination_number,4)
    
    @property
    def bond_length(self):
        if self._bond_length is None:
            all_distance = []
            cnn = CrystalNN()
            
            for i in range(len(self._structure)):
                nn_info = cnn.get_nn_info(self._structure , i)

                for neighbor in nn_info:
                    dist = self._structure[i].distance(neighbor["site"])
                    all_distance.append(dist)
            self._bond_length = np.mean(all_distance)

        return round(self._bond_length,4)
    
    @property
    def bond_length_min(self):
        if self._bond_length_min is None:
            all_distance = []
            cnn = CrystalNN()
            
            for i in range(len(self._structure)):
                nn_info = cnn.get_nn_info(self._structure , i)

                for neighbor in nn_info:
                    dist = self._structure[i].distance(neighbor["site"])
                    all_distance.append(dist)
            self._bond_length_min = min(all_distance)

        return round(self._bond_length_min,4)
    
    @property
    def bond_length_max(self):
        if self._bond_length_max is None:
            all_distance = []
            cnn = CrystalNN()
            
            for i in range(len(self._structure)):
                nn_info = cnn.get_nn_info(self._structure , i)

                for neighbor in nn_info:
                    dist = self._structure[i].distance(neighbor["site"])
                    all_distance.append(dist)
            self._bond_length_max = max(all_distance)

        return round(self._bond_length_max,4)
    
    @property
    def bond_length_std(self):
        if self._bond_length_std is None:
            cnn = CrystalNN()
            site_std = []

            for i in range(len(self._structure)):
                nn_info = cnn.get_nn_info(self._structure , i)
                local_distances = []

                for neighbor in nn_info:
                    dist = self._structure[i].distance(neighbor["site"])
                    local_distances.append(dist)

                if local_distances:
                    site_std.append(np.std(local_distances))

            self._bond_length_std = np.mean(site_std)
        return round(self._bond_length_std,4)
    
    @property
    def packing_fraction(self):
        if self._packing_fraction is None:

            def volume(r):
                return (4.0/3.0) * (np.pi) * (r ** 3)
            
            atomic_volume = 0.0
            for site in self._structure:
                radius = site.specie.atomic_radius
                if radius is None:
                    radius = site.specie.covalent_radius
                atomic_volume += volume(radius)
            cell_volume = self.cell_volume
            self._packing_fraction = atomic_volume / cell_volume

        return round(self._packing_fraction,4)
    
    @property
    def bond_angle_mean(self):
        if self._bond_angle_mean is None:
            cnn = CrystalNN()
            angles = []

            for center in range(len(self._structure)):
                neighbors = cnn.get_nn_info(self._structure,center)
                neigh_idx = [n["site_index"] for n in neighbors]

                for i,j in combinations(neigh_idx,2):
                    angle = self._structure.get_angle(i,center,j)
                    angles.append(angle)
            if not angles:
                self._bond_angle_mean = 0.0
            else:
                self._bond_angle_mean = np.mean(angles)

        return round(self._bond_angle_mean,4)
    
    @property
    def bond_angle_std(self):
        if self._bond_angle_std is None:
            cnn = CrystalNN()
            angles = []

            for center in range(len(self._structure)):
                neighbors = cnn.get_nn_info(self._structure,center)
                neigh_idx = [n["site_index"] for n in neighbors]

                for i,j in combinations(neigh_idx,2):
                    angle = self._structure.get_angle(i,center,j)
                    angles.append(angle)
            if not angles:
                self._bond_angle_std = 0.0
            else:
                self._bond_angle_std = np.std(angles)

        return round(self._bond_angle_std,4)
    
    @property
    def econ(self):
        if self._econ is None:
            econ_class = EconNN()
            econs = []

            for i in range(len(self._structure)):
                try:
                    nn_info = econ_class.get_nn_info(self._structure,i)
                    econs.append(sum(nn["weight"] for nn in nn_info))
                except:
                    pass

            self._econ = np.mean(econs)

        return round(self._econ,4)
    
    @property
    def valence_electron_count(self):
        if self._valence_electron_count is None:
            val_electrones = 0.0
            el_atm_dic = self._structure.composition.get_el_amt_dict()
            total_atoms = self._structure.composition.num_atoms

            for symbol,amount in el_atm_dic.items():
                element = Element(symbol)
                highest_n = max(n for n,_,_ in element.full_electronic_structure)
                val_ele =sum(
                    occ
                    for n,_,occ in element.full_electronic_structure
                    if n == highest_n
                )
                val_electrones += val_ele * amount
            
            self._valence_electron_count = val_electrones / total_atoms
        return round(self._valence_electron_count,4)
    
    @property
    def d_electron_count(self):
        if self._d_electron_count is None:
            self._d_electron_count = {}
            elements = self._structure.elements

            for ele in elements:
                if ele.is_transition_metal:
                    d_electron = 0
                    for n,orb,occ in ele.full_electronic_structure:
                        if orb == "d":
                            d_electron += occ
                    self._d_electron_count[ele.symbol] = d_electron

            if len(self._d_electron_count) == 0:
                self._d_electron_count = None

        return self._d_electron_count
    
    @property
    def bandgap_estimate(self):
        if self._bandgap_estimate is None:
            elements = self._structure.elements
            has_metal = any(ele.is_metal for ele in elements)

            if has_metal:
                self._bandgap_estimate = (0,4)
            else:
                self._bandgap_estimate = (2,10)

        return self._bandgap_estimate
    
    @property
    def bandgap_ml(self):
        if self._bandgap_ml is None:
            model = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")
            state_attr = torch.tensor([0], dtype=torch.long)
            prediction = model.predict_structure(self._structure, state_attr=state_attr)
            self._bandgap_ml = prediction.item()
        
        return round(self._bandgap_ml,4)
    
    @property
    def electronic_type(self):
        if self._electronic_type is None:
            band_gap = self.bandgap_ml
            
            if band_gap <=0 :
                self._electronic_type = "metal"
            elif 0 < band_gap < 0.1:
                self._electronic_type = "semimetal"
            elif 0.1 < band_gap < 3:
                self._electronic_type = "semiconductor"
            else:
                self._electronic_type = "insultor"

        return self._electronic_type
    
    @property
    def avg_ionization_energy(self):
        if self._avg_ionization_energy is None:
            composition = self._structure.composition
            ie = 0
            total_atoms = composition.num_atoms

            for ele,amt in composition.items():
                ie += Element(ele.symbol).ionization_energy * amt

            self._avg_ionization_energy = round(ie / total_atoms,4)

        return self._avg_ionization_energy

    
    def get_neighbors(self,site_idx):
        cnn = CrystalNN()
        nn_info = cnn.get_nn_info(self._structure,site_idx)
        neighbors_list = []

        for n in nn_info:
            n_site = n["site"]
            info_dict = {}
            info_dict["element"] = n_site.specie
            info_dict["distance"] = float(round(n_site.distance(self._structure[site_idx]),4))
            info_dict["site_idx"] = int(n_site.index)
            neighbors_list.append(info_dict)

        return neighbors_list
    
    def compare(self, other):
        result = {}
        Properties = [
            #identity
            "formula",
            #structure
            "crystal_system","space_group","space_group_number","n_symmetry_ops","is_centrosymmetric","is_polar",
            #lattice
            "a","b","c","alpha","beta","gamma","cell_volume","density","n_atoms_unit_cell",
            #environment
            "coordination_number","bond_length","bond_length_min","bond_length_max","bond_length_std","packing_fraction","bond_angle_mean",
            #composition
            "electronegativity_diff","ionic_character","valence_electron_count","avg_atomic_mass",
            #electronic
            "bandgap_ml","electronic_type","avg_ionization_energy"
        ]
        Categorical = {
            "formula","crystal_system","space_group","electronic_type"
        }
        Boolean = {
            "is_centrosymmetric" , "is_polar"
        }
        for prop in Properties:
            v1 = getattr(self,prop)
            v2 = getattr(other,prop)

            if prop in Categorical or prop in Boolean:
                result[prop] = {
                    "mat1" : v1, "mat2" : v2,"same" : v1 == v2
                }
            else:
                result[prop] = {
                    "mat1" : v1, "mat2" : v2,"diff" : round(v2-v1,4)
                }
        return result

