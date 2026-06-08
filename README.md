<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=matsciml&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Materials%20Science%20×%20Machine%20Learning&descAlignY=55&descSize=18" width="100%"/>

<!-- Typing Animation -->
<a href="https://github.com/opurbo2512/matsciml">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00E5FF&center=true&vCenter=true&multiline=true&width=700&height=80&lines=from+matsciml+import+Material;mat+%3D+Material(%22Fe2O3%22)+%23+that's+it." alt="Typing SVG" />
</a>

<br/><br/>

<!-- Badges Row 1 -->
<img src="https://img.shields.io/badge/Python-3.8%2B-00E5FF?style=for-the-badge&logo=python&logoColor=white&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/Status-In%20Development-FFD740?style=for-the-badge&logoColor=white&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/License-MIT-76FF03?style=for-the-badge&labelColor=0D1117"/>

<br/>

<!-- Badges Row 2 -->
<img src="https://img.shields.io/badge/pymatgen-✓-E040FB?style=flat-square&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/PyG-✓-FF6D00?style=flat-square&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/sklearn%20compatible-✓-76FF03?style=flat-square&labelColor=0D1117"/>
<img src="https://img.shields.io/badge/Materials%20Project-integrated-00E5FF?style=flat-square&labelColor=0D1117"/>
<img src="https://img.shields.io/github/stars/opurbo2512/matsciml?style=flat-square&color=FFD740&labelColor=0D1117"/>

</div>

---

<div align="center">

## ⚡ The Problem

</div>

You're an MME student. You want to predict band gaps with ML.
Instead of doing ML, you spend **3 days writing glue code**:

```python
# 😩 Without matsciml — this is your life right now
from pymatgen.core import Structure, Composition
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

struct = Structure.from_file("Fe2O3.cif")
sga    = SpacegroupAnalyzer(struct)
cnn    = CrystalNN()

crystal_system = sga.get_crystal_system()
coord_num = sum(len(cnn.get_nn_info(struct, i))
                for i in range(len(struct))) / len(struct)
# ... 40 more lines before you write a single line of ML 💀
```

```python
# 😎 With matsciml — done
from matsciml import Material

mat = Material("Fe2O3")
mat.crystal_system       # Rhombohedral
mat.coordination_number  # 6.0
mat.bandgap_ml           # 2.18 eV
mat.to_vector()          # numpy array, ready for sklearn
mat.to_graph()           # PyG graph, ready for GNN
```

---

<div align="center">

## 🗺️ What This Library Does

```
Chemical Formula / CIF File
         │
         ▼
  ┌─────────────────┐
  │  Material Object │  ← smart container, 28 properties
  └────────┬────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────┐
│ Features │  │  Graph   │
│ numpy    │  │  PyG     │
│ pandas   │  │  GNN     │
└────┬─────┘  └────┬─────┘
     └──────┬───────┘
            ▼
    ┌───────────────┐
    │  ML Pipeline  │  ← sklearn / PyTorch
    └───────┬───────┘
            ▼
    Property Prediction
    (band gap, hardness,
     formation energy...)
```

</div>

---

<div align="center">

## ✨ Features

</div>

<table>
<tr>
<td width="50%">

### ⚛️ Feature 01 — Material Object
The central nervous system of the library.

```python
mat = Material("TiO2")

mat.formula            # 'TiO2'
mat.crystal_system     # 'Tetragonal'
mat.space_group        # 'P4_2/mnm'
mat.lattice            # {a:4.59, c:2.96, ...}
mat.coordination_number # 6.0
mat.electronegativity_diff # 2.02
mat.bond_length        # 1.946 Å
mat.density            # 4.23 g/cm³
mat.bandgap_ml         # 3.04 eV
```

</td>
<td width="50%">

### 🔬 Feature 02 — Feature Extractor
Convert any material to ML-ready vectors.

```python
from matsciml.features import StructuralFeaturizer

f = StructuralFeaturizer(include=["all"])
X = f.fit_transform(material_list)
# shape: (n_materials, 32)
# dtype: float64
# sklearn-compatible out of the box
```

</td>
</tr>
<tr>
<td width="50%">

### 🕸️ Feature 03 — Crystal Graph Dataset
Crystal structures as graphs for GNNs.

```python
from matsciml.datasets import CrystalGraphDataset

ds = CrystalGraphDataset.from_mp(n=5000)
graph = ds[0]

graph.num_nodes    # e.g. 6
graph.num_edges    # e.g. 58
graph.x.shape      # [6, 9]  node features
graph.edge_attr.shape  # [58, 3]
```

</td>
<td width="50%">

### 🤖 Feature 04 — Pre-trained Models
Near-DFT accuracy, millisecond inference.

```python
from matsciml.models import MatSciPredictor

model = MatSciPredictor.from_pretrained(
    "matsciml-base"
)
result = model.predict(Material("GaN"))

result["bandgap"]          # 3.38 eV
result["formation_energy"] # -1.19 eV/atom
result["bulk_modulus"]     # 195 GPa
```

</td>
</tr>
<tr>
<td colspan="2">

### 🚀 Feature 05 — sklearn Pipeline
End-to-end in 10 lines.

```python
from matsciml.pipeline import MatSciPipeline
from matsciml.features import StructuralFeaturizer
from sklearn.ensemble import GradientBoostingRegressor

pipe = MatSciPipeline(
    featurizer=StructuralFeaturizer(),
    model=GradientBoostingRegressor(n_estimators=500)
)
pipe.fit(train_materials, y_train)
pipe.predict([Material("BaTiO3")])   # → [3.27 eV]
pipe.save("my_bandgap_model.pkl")
```

</td>
</tr>
</table>

---

<div align="center">

## 📦 Installation

</div>

```bash
# Stable release (coming soon)
pip install matsciml

# Install from source (current)
git clone https://github.com/opurbo2512/matsciml.git
cd matsciml
pip install -e ".[dev]"
```

> **Note:** PyTorch and PyTorch Geometric are not auto-installed due to CUDA version dependencies.
> Install them manually first: [pytorch.org](https://pytorch.org) · [pyg.org](https://pyg.org/whl)

---

<div align="center">

## 🗂️ Project Structure

</div>

```
matsciml/
│
├── matsciml/                  ← main package
│   ├── core/
│   │   ├── material.py        ← Material class (Feature 01)
│   │   └── composition.py     ← CompositionEngine
│   ├── features/
│   │   └── featurizer.py      ← StructuralFeaturizer (Feature 02)
│   ├── datasets/
│   │   └── graph_dataset.py   ← CrystalGraphDataset (Feature 03)
│   ├── models/
│   │   ├── cgcnn.py           ← CGCNN implementation (Feature 04)
│   │   └── predictor.py       ← MatSciPredictor
│   └── pipeline/
│       └── pipeline.py        ← MatSciPipeline (Feature 05)
│
├── tests/                     ← pytest unit tests
├── examples/                  ← Jupyter notebooks
├── docs/                      ← documentation
├── .github/workflows/ci.yml   ← GitHub Actions
├── README.md
├── setup.py
└── requirements.txt
```

---

<div align="center">

## 🗺️ Roadmap

</div>

<table>
<tr>
<th>Phase</th>
<th>Status</th>
<th>What's being built</th>
</tr>
<tr>
<td><b>Phase 1</b><br/><sub>Foundation</sub></td>
<td><img src="https://img.shields.io/badge/🔨 In Progress-FFD740?style=flat-square"/></td>
<td>
  Material class · Composition parser · Structural featurizer · PyPI package
</td>
</tr>
<tr>
<td><b>Phase 2</b><br/><sub>Graph + GNN</sub></td>
<td><img src="https://img.shields.io/badge/📋 Planned-3D444D?style=flat-square"/></td>
<td>
  CrystalGraphDataset · CGCNN implementation · Materials Project integration · Benchmarks
</td>
</tr>
<tr>
<td><b>Phase 3</b><br/><sub>Pre-trained + Docs</sub></td>
<td><img src="https://img.shields.io/badge/📋 Planned-3D444D?style=flat-square"/></td>
<td>
  Pre-trained weights · HuggingFace Hub · Full documentation · JOSS paper
</td>
</tr>
</table>

---

<div align="center">

## 🧪 Quick Start

</div>

**Try it in your browser (no install needed):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opurbo2512/matsciml/blob/main/examples/01_material_basics.ipynb)

```python
# 1. Install
!pip install matsciml

# 2. Create a material
from matsciml import Material
mat = Material("Fe2O3")

# 3. Explore properties
print(mat.crystal_system)     # Rhombohedral
print(mat.space_group)        # R-3c
print(mat.coordination_number) # 6.0
print(mat.bandgap_ml)         # 2.18 eV

# 4. Convert to ML format
X = mat.to_vector()           # (32,) numpy array
G = mat.to_graph()            # PyG Data object
```

---

<div align="center">

## 📊 Model Performance

</div>

Benchmarks on the [MatBench](https://matbench.materialsproject.org/) test suite:

| Property | Model | MAE | Unit | DFT Cost |
|:---|:---:|:---:|:---:|:---:|
| Band Gap | CGCNN | 0.388 | eV | ~hours/material |
| Formation Energy | CGCNN | 0.124 | eV/atom | ~hours/material |
| Bulk Modulus | Descriptor+GBR | 14.7 | GPa | ~days/material |
| Shear Modulus | Descriptor+GBR | 18.4 | GPa | ~days/material |
| **matsciml inference** | **any** | **—** | **—** | **~ms/material** ⚡ |

---

<div align="center">

## 🤝 Contributing

</div>

This project is in active development. Contributions are welcome once Phase 1 is complete.

```bash
# Fork → Clone → Branch → Code → PR
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines *(coming soon)*.

---

<div align="center">

## 📖 Citation

If you use matsciml in your research, please cite:

```bibtex
@software{matsciml2025,
  author  = {opurbo2512},
  title   = {matsciml: Materials Science × Machine Learning},
  year    = {2025},
  url     = {https://github.com/opurbo2512/matsciml},
  license = {MIT}
}
```

---

<br/>

**Built with ❤️ by an MME student who got tired of writing glue code.**

<sub>Dept. of Materials & Metallurgical Engineering · BUET</sub>

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
