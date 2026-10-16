# Victor_Turyahebwa_B40815_Assignment_1_Advent_2026_MiniProjects
Assignment 1 OOP Mini Projects

## Repository structure

```text
.
├── project1_population.ipynb
├── project2_microgrid.ipynb
├── project3_fishery.ipynb
├── project4_rainfall.ipynb
├── project5_taxi.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── microgrid_30_days.csv
├── src/
│   ├── __init__.py
│   ├── fishery.py
│   ├── microgrid.py
│   ├── population.py
│   ├── rainfall.py
│   └── taxi.py
└── tests/
    ├── test_fishery.py
    ├── test_microgrid.py
    ├── test_population.py
    ├── test_rainfall.py
    └── test_taxi.py
```

## Folder layout

| Notebook | Reusable module | Main result |
| --- | --- | --- |
| `project1_population.ipynb` | `src/population.py` | Train/test model scores; 2025–2029 projections and extra classroom estimates |
| `project2_microgrid.ipynb` | `src/microgrid.py` | Thirty-day dispatch, NNLS feasibility flags and monthly illustrative cost |
| `project3_fishery.ipynb` | `src/fishery.py` | Fish-stock/revenue scenarios and simulated downside quantile |
| `project4_rainfall.ipynb` | `src/rainfall.py` | Monthly crop-moisture screening and region comparisons |
| `project5_taxi.ipynb` | `src/taxi.py` | Fare equilibrium, day-11 revenue forecasts and buffered vehicle counts |

All `src` modules use plain classes and methods, straightforward validation, short functions and documented units. Abstract base classes appear only where the brief explicitly asks for related forecast subclasses. `tests/` contains at least three normal/edge-case tests for **each** mini-project. Every notebook ends with a 150–300-word Findings & Limitations section.


## Install and run

From the extracted repository folder, in a terminal with Python 3.10 or later:

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows PowerShell; `source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m jupyter lab
```

Open each notebook from the main repository folder. Choose **Restart Kernel and Run All Cells**. The micro-grid notebook has `RUN_INTERACTIVE = False` to keep this action non-blocking; change it to `True` to try the validated `input()` mode. The notebook generates/replaces `data/microgrid_30_days.csv` with the same seed every run. The generated CSV is also included for inspection. If you launch Jupyter inside the `notebooks` directory (there is none in this repository), return to the main folder so `src` can be imported.

## Findings at a glance

- **Population:** CAGR scored best by held-out MAE for all five illustrative series. Wakiso has the fastest relative growth. Estimated additional classrooms by 2029 are based only on projected growth since 2024, not on the total number already needed.
- **Micro-grid:** The determinant is −5, so the equations have a unique algebraic solution. Two of the 30 deliberately constructed days require non-negative least squares, with the remaining mismatch reported instead of concealed.
- **Fishery:** The logistic model has a theoretical equilibrium MSY of 1,000 tonnes per week, but the actual 52-week starting stock is below the 5,000-tonne MSY stock. Revenue variance must be stated in UGX²; CV and a simulated 5th percentile add scale-aware context.
- **Rainfall:** The illustrative Kampala profile yields only one detected peak, unlike the typical southern bimodal pattern. Gulu yields two shoulders under the specified peak rule, consistent with a district-level climate description. Replace illustrative rainfall before providing real advice.
- **Taxi:** The assumed curves give a Ntinda equilibrium fare of UGX 2,200. All illustrated day-11 passenger forecasts fit into one 14-seat vehicle making eight trips when a 15% buffer is added. Profit cannot be inferred without costs.


## AI-use declaration

An AI coding assistant was used to help structure the repository, refine the reports, refine the code for better efficiency, discussions and documentation.
