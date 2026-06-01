# F1 Session Intelligence Dashboard

No-code FastF1 dashboard for comparing drivers and teams by circuit section.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar to select the season, race, and session. The app loads FastF1
timing and telemetry, then builds:

- corner team and driver rankings
- fastest-driver track heat map by mini-sector
- straight team and driver rankings
- minimum-speed corner leaderboard
- speed-trap leaderboard
- best theoretical lap leaderboard
- long-run pace simulation leaderboard

FastF1 caches data in `fastf1_cache/` so repeated loads are much faster.
