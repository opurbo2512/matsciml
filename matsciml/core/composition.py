from pymatgen.core import Composition

class CompositionEngine:
    def __init__(self,formula):
        self.formula = formula
        self._element_ratios = None

    @property
    def element_ratios(self):
        if self._element_ratios is None:
            comp = Composition(self.formula)
            self._element_ratios = comp.get_el_amt_dict()
        return self._element_ratios
    
    

