from pymatgen.core import Composition
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from mp_api.client import MPRester
from pathlib import Path

class Material:
    def __init__(self,input_formula):
        self.formula = input_formula
        self._density = None
        self.structure = None
        self.sga = None
        if self.formula is not None:
            self._run_initial()

    def _run_initial(self):
        self.formula = self.formula.replace(" ","")
        API_KEY = "ihddElrSgFw4L4DlUUb4VkH1px5SYhS0"
        with MPRester(API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula = self.formula
            )
        best = min(
            docs,
            key = lambda x : x.energy_above_hull
        )
        self.structure = best.structure
        self.sga = SpacegroupAnalyzer(self.structure)

    @classmethod
    def from_cif(cls,file_name):
        instance = cls(input_formula = None)
        instance.structure = Structure.from_file(file_name)
        instance.sga = SpacegroupAnalyzer(instance.structure)

        return instance
    
    @classmethod
    def from_poscar(cls,file_name):
        instance = cls(input_formula = None)
        instance.structure = Structure.from_file(file_name)
        instance.sga = SpacegroupAnalyzer(instance.structure)

        return instance

    
    @classmethod
    def from_mp(cls,mp_id):
        instance = cls(input_formula = None)
        API_KEY = "ihddElrSgFw4L4DlUUb4VkH1px5SYhS0"
        with MPRester(API_KEY) as mpr:
            instance.structure = mpr.get_structure_by_material_id(mp_id)
        instance.sga = SpacegroupAnalyzer(instance.structure)

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
        instance.structure = Structure(lattice,species,coords)
        instance.sga = SpacegroupAnalyzer(instance.structure)

        return instance
            

    @property
    def density(self):
        if self._density is None:
            self._density = self.structure.density
        return self._density
    
