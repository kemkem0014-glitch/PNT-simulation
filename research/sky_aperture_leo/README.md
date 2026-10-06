# Sky-Aperture Geometry for Single-LEO Doppler Positioning

This subproject studies intermittent single-satellite LEO Doppler positioning under obstructed sky conditions.

## Question

With the same total observation time or the same total angular amount of open sky, how do the **number, size, and placement of separated sky apertures** affect positioning observability and accuracy?

The hypothesis is that a fast-moving LEO converts separated physical openings into **temporal geometric diversity**, so several well-separated small openings can outperform one continuous opening.

## Baseline model

- stationary receiver with coarse prior position
- one LEO satellite, altitude 550 km
- closest ground-track offset 300 km
- 20 deg elevation mask
- representative carrier 2 GHz
- range-rate noise 0.2 m/s (~1.33 Hz at 2 GHz)
- state: horizontal position x/y + common range-rate/frequency bias
- 1 Hz samples available only inside apertures

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python aperture_leo_sim.py --output-dir output_aperture_leo
```

The script generates CRLB/FIM optimization results, random-layout comparisons, Monte Carlo validation, CSVs, and figures.

See [REPORT.md](REPORT.md) for the current simulation study.

## Status

This is an idealized geometry paper model. Follow-on work should replace the local pass with TLE/SGP4 and add C/N0/elevation dependence, oscillator dynamics, reacquisition time, real signal structure, and measured forest sky masks.
