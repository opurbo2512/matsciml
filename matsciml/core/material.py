from pymatgen.core import Composition
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer

from mp_api.client import MPRester
from composition import CompositionEngine
import math

class Material:
    def __init__(self,input_formula):
        self.input_formula = input_formula
        self._structure = None
        self._sga = None

        self._formula = None
        self._name = None #do this later
        self._elements = None
        self._n_elements = None
        self._source = None
        self._composition = None
        self._electronegativity_diff = None
        self._oxidation_states = None # there are some bug, should update later
        self._valence_electrons = None
        self._ionic_character = None
        self._crystal_type = None # do this later
        self._crystal_system = None
        self._space_group = None
        self._space_group_number = None
        self._lattice = None
        self._cell_volume = None
        self._density = None
        self._n_atoms_unit_cell = None
        self._n_symmetry_ops = None
        self._is_centrosymmetric = None
        self._is_polar = None
        
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
    
    @property
    def name(self):
        if self._name is None:
            pass
        return self._name

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
                valenc_cofig = element.valence
                valence_electrons = sum(valenc_cofig)
                self._valence_electrons[symbol] = valence_electrons

        return self._valence_electrons
    
    @property
    def ionic_character(self):
        if self._ionic_character is None:
            en_diff = self.electronegativity_diff
            self._ionic_character = (1-math.exp(-((en_diff ** 2)/4))) * 100

        return round(self._ionic_character,4)
    
    @property
    def crystal_type(self):
        if self._crystal_type is None:
            pass

        return self._crystal_type
    
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
    
