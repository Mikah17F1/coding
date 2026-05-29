# F1 qualifying speed delta

Create a qualifying development graph by race with FastF1.

```bash
python -m pip install -r requirements.txt
python qualifying_speed_delta.py --season 2024
```

The script uses each team's fastest qualifying result from Q1/Q2/Q3, compares it
to the benchmark for that race, and writes:

- `outputs/quali_speed_delta_<season>.png`
- `outputs/quali_speed_delta_<season>.csv`

Useful options:

```bash
# Plot a subset of teams
python qualifying_speed_delta.py --season 2024 --teams "Red Bull,Ferrari,McLaren"

# Smoke-test one part of a season
python qualifying_speed_delta.py --season 2024 --start-round 1 --end-round 5

# Compare everyone to one team instead of the fastest team each race
python qualifying_speed_delta.py --season 2024 --benchmark-team McLaren

# Plot seconds instead of speed deficit percent
python qualifying_speed_delta.py --season 2024 --metric time-delta

# Render a TV-style 2025 graphic
python qualifying_speed_delta.py --season 2025 --output outputs/quali_speed_delta_2025_tv.png --csv outputs/quali_speed_delta_2025.csv
```
