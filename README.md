# Beyond Activation Maps: Higher-Order Information and Temporal Topology of Decodable Cognitive States in fMRI


For each brain state class `c`, the analysis compares:

- `M_c`: significant regions.
- `L_c`: low-weight regions.

Each regional BOLD time series is standardized, embedded with Takens delay
coordinates, converted into a Vietoris-Rips filtration, and summarized by the
finite `H1` persistence diagram `D_1`: the number of loops, total persistence
`P(D_1)`, and maximal lifetime `M(D_1)`.

## Requirements

Use Python 3.10 or newer. Install the dependencies into the Jupyter kernel
environment that will run the notebook:

```setup
pip install -r requirements.txt
```

## Data

Expected folders inside the downloaded folder:

```text
data_Regions_with_low_weights/
data_Significant_regions/
```

Expected file layout:

```text
<region-set-root>/<EXPERIMENT>/<subject_id>_<phase_encoding>_<state>.csv
```

The first column contains region indices. Remaining columns are ordered BOLD
time points. For example, `100206_RL_math.csv` denotes subject `100206`, RL
phase encoding, and the LANGUAGE Math state.

State names used by this release:

| Experiment | States |
| --- | --- |
| LANGUAGE | `math`, `story` |
| MOTOR | `l`, `r` |
| SOCIAL | `mental`, `rnd` |

## Repository Structure

```text
<downloaded-folder>/
  README.md
  requirements.txt
  region_persistence.py
  region_persistence_batch_analysis.ipynb
  examples/mini_data/
```

`region_persistence.py` contains in-memory Takens embedding, persistence,
aggregation, surrogate, and figure-construction functions. 

`region_persistence_batch_analysis.ipynb` is the main notebook for reproducing
the tables and figures. It owns all file management: input folders, output
folders, cache validation, metadata writing, and figure saving.

`examples/mini_data/` is a synthetic smoke-test fixture. It is not used for the
paper results.

## Citation

#TBD

## Contributing
