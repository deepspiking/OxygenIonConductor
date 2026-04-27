Information on the stored data
------------------------------

**Main database: see database/oxygen_ion_conductor_dataset.csv**

| Item | Description |
|---|---|
| References | Family name of the first author, journal title, volume, initial page number, and published year. |
| DOI | Digital Object Identifier of the source publication, enabling traceability. |
| Year | Publication year of the experimental report. |
| Class | Structural class of the material (e.g., perovskite, scheelite, apatite). |
| Formula | Reported chemical composition. |
| Parsed formula | Standardized composition represented by A_s–B_s(–C_s) site classification. For example, in the perovskite oxide La₀.₉₅Sr₀.₀₅Ga₀.₉₅Mg₀.₀₅O₃₋δ, La and Sr occupy the A_s-site, while Ga and Mg occupy the B_s-site of the perovskite lattice. |
| Eₐ / A | Activation energy in meV and prefactor in K·S·cm⁻¹. When there are two distinct linear regions, values for the low-temperature regime are shown. |
| Eₐ,HT / A_HT | Activation energy and prefactor values for the high-temperature regime (if applicable). |
| T* | Transition temperature separating low- and high-temperature regimes (if applicable). |
| Type of σ_T | Conductivity type, distinguishing bulk conductivity from total conductivity (bulk and grain boundary contributions). |
| Measurement | Experimental technique employed (e.g., two-probe AC, impedance spectroscopy). |
| Measurement temperature range | The range of temperatures over which experimental measurements were performed. |
| Source | Origin of the data within the publication (figure or table reference). |
| Plot type | Axis configuration used in the original Arrhenius representation (e.g., ln σ_T – 1000/T). |

**[Reference]**

Note. Will be uploaded soon in arXiv.

Regression model
----------------
**Symbolic regression modelling package: see modelling_GoodRegressor**
## How to Compile

### Example

```bash
g++ GoodDesigner.cpp -o GoodDesigner.x -std=c++11
g++ GoodCurator.cpp  -o GoodCurator.x  -std=c++11
mpicxx GoodRegressor_slow.cpp -o GoodRegressor.x -std=c++11
```

### Notes

- **GoodDesigner** and **GoodRegressor** require the **Eigen** library  
  (see: https://libeigen.gitlab.io/).  
  If you use a custom Eigen installation, add:
  ```bash
  -I/path/to/Eigen
  ```

- As the filename suggests, there exists a **faster version** of `GoodRegressor`
  (approximately **100× faster** on a node-hour scale),
  which is **not yet publicly available**.

---

## Directory Overview

Each directory contains input files and a source file  
(the same source files are also provided in `CodeOnly`).  
Output files are stored separately in `ref_output`.

### Modules

- **001_GoodDesigner**  
  From compositions and structures, this module extracts features
  (descriptor candidates).

- **002_GoodCurator**  
  This module adds first-order (simple) interactions between feature terms.

- **003_GoodRegressor**  
  This module generates symbolic regression models.  
  In this example, it produces:
  - 20 symbolic regression models for **Ea**
  - 10 symbolic regression models for **log₁₀A**

- **004_GoodDesignerPost**  
  This module generates the final stacking-ensembled models and identifies
  important features and interactions.  
  It also generates **LaTeX** and **Mathematica** code for the symbolic
  regression formulae for further post-processing by users.

Reference
---------
Jang, S.-H., Kiyohara, S., Takamura, H. & Kumagai, Y. Charting the Landscape of Oxygen Ion Conductors: A 60-Year Dataset with Interpretable Regression Models. Sci Data (2026). https://doi.org/10.1038/s41597-026-07100-x

Jang, S.-H. GoodRegressor: A Hierarchical Inductive Bias for Navigating High-Dimensional Compositional Space, arXiv:2510.18325 
https://doi.org/10.48550/arXiv.2510.18325
