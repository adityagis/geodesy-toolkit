# geodesy-toolkit

Small Python tools for the checks I run on GNSS and levelling data.

- **`rinex_qc`** checks RINEX observation files (2.x and 3.x): station, receiver, antenna, interval, first and last epoch, epoch count, completeness and data gaps. Point it at a folder to check a whole day or year of files, and save the result as CSV.
- **`levelling`** adjusts a levelling network by least squares and runs the checks that catch bad observations: the global chi-square test on the variance factor and Pope's Tau test on each residual. It also has helpers for geopotential numbers and Helmert orthometric heights.

The example data in `examples/` is synthetic. No survey data from any organisation is included.

Maintained by Aditya Kumar Jha. For scoped GNSS processing and survey QA work, see [Uday GeoAnalytics Consultancy](https://ugaconsult.com/gnss-survey-qa).

## Install

```bash
git clone https://github.com/adityagis/geodesy-toolkit.git
cd geodesy-toolkit
pip install -e .
```

Needs Python 3.9+, NumPy and SciPy.

## RINEX quality check

```bash
python -m geodesy_toolkit.rinex_qc examples
python -m geodesy_toolkit.rinex_qc /data/rinex/2026 --csv qc_report.csv
```

```
dem22440.26o                 DEM2       epochs=120    interval=30s completeness= 100.0% gaps=0
DEMO00IND_R_20262440000_01H_30S_MO.rnx DEMO       epochs=110    interval=30s completeness=  91.7% gaps=1
```

Completeness is the number of epochs found divided by the number expected between the first and last epoch at the file's interval. A gap is any jump longer than 1.5 times the interval. Plain and `.gz` files are both read.

## Levelling network adjustment

Two CSV files:

```
observations.csv                  fixed.csv
from,to,dh_m,length_km            point,height_m
BM1,A,2.34500,1.8                 BM1,100.0000
A,B,2.77546,2.4                   BM2,104.5000
...
```

```bash
python -m geodesy_toolkit.levelling examples/levelling_observations.csv examples/levelling_fixed.csv --sigma 1.0
```

```
Observations: 9   unknown heights: 4   degrees of freedom: 5
Variance factor: 17.282
Global test: chi2 = 86.41, accepted range [0.83, 12.83] -> FAILED

Adjusted heights
  A                102.3453 m   +/-   3.56 mm
  ...

Tau test (critical value 2.10)
  ...
         B -> C        v =  -10.49 mm   tau =  -2.21  <-- check
```

The example has a 15 mm blunder put in on purpose on the line B → C. The global test fails and the Tau test points to that line.

How it works:

- Each observation gets a standard deviation of `sigma × √length_km` (mm), and weight `1/σ²`.
- Heights are solved with the normal equations `x = (AᵀPA)⁻¹ AᵀPl`.
- Global test: `vᵀPv` is compared with the chi-square bounds for the degrees of freedom.
- Tau test: each residual is divided by `σ̂₀ √q_vv`. The critical value comes from the Tau distribution with a Bonferroni correction over all observations.

Helper functions:

```python
from geodesy_toolkit.levelling import geopotential_differences, helmert_height

dC = geopotential_differences(dh_m=[1.234, -0.567], mean_gravity_gal=[979.12, 979.10])  # gpu
H = helmert_height(geopotential_number_gpu=980.0, surface_gravity_gal=979.0)            # metres
```

## Tests

```bash
pip install pytest
pytest
```

## About

Written by [Aditya Kumar Jha](https://adityagis.github.io/Portfolio/), GIS engineer at Survey of India's Geodetic & Research Branch. Built with AI-assisted development and checked against known results. MIT licence.
