# DC Bikeshare Mobility Explorer — DATS6401 Midterm

## Research question
When do people ride, where is station activity concentrated, and which stations are most strongly connected?

## Three data modalities
1. **Time series:** trips per day and start hour from timestamp fields.
2. **Geographic map:** station locations, sized by recorded activity.
3. **Network:** directed station-to-station rides aggregated for a readable node-link diagram, degree sizing and modularity-community coloring.

## Source and period
Public Capital Bikeshare trip history: https://capitalbikeshare.com/system-data
Default: **April 2025** (`202504`). Download ZIP from:
https://s3.amazonaws.com/capitalbikeshare-data/202504-capitalbikeshare-tripdata.zip
The files cover Washington, DC and surrounding municipalities. The visualizations show historical trip records, not live availability.

## Files
- `prepare_midterm_data.py`: download and preprocess a month of official records.
- `midterm_bikeshare_app.py`: Streamlit app containing all tabs.
- `requirements_midterm_additions.txt`: package dependencies; add missing lines to your existing **root** `requirements.txt` rather than replacing its earlier course requirements.
- `data/bikeshare_midterm/*.csv` and `metadata.json`: generated reproducible data. **Commit these small derived files to GitHub** so the deployed app never needs to fetch large ZIP files.

## Run on your Mac
From your `DATS6401` folder:
```bash
conda activate dataviz
python -m pip install -r requirements_midterm_additions.txt
python prepare_midterm_data.py
python -m streamlit run midterm_bikeshare_app.py
```
You may choose a different month using `python prepare_midterm_data.py --month 202505` (this REPLACES the prepared month rather than combining months).

## What should appear?
- Overview: totals (all valid trips vs station-to-station trips differ deliberately).
- Time series: daily counts and start hour counts.
- Station map: actual station positions and observed activity.
- Station network: choose stations, trip threshold, and layout; network size is degree and color is greedy-modularity community.
- Findings & methods: descriptions, limitations, accessibility notes, source and preprocessing counts.

## Deploy on Streamlit Community Cloud
1. Ensure root `requirements.txt` contains `streamlit`, `pandas`, `numpy`, `plotly`, `networkx`, and `requests` (the latter is only required for the downloader locally; harmless on Cloud). Preserve prior projects' dependencies.
2. Confirm `data/bikeshare_midterm/daily_trips.csv`, `hourly_trips.csv`, `daily_routes.csv`, `stations.csv` and `metadata.json` exist.
3. Push:
```bash
git add midterm_bikeshare_app.py prepare_midterm_data.py README_MIDTERM.md requirements.txt requirements_midterm_additions.txt data/bikeshare_midterm/
git commit -m "Add DATS6401 multimodal bikeshare midterm"
git push
```
4. Create a **new Streamlit Cloud app** with repo `Simba-gwu/DATS6401`, branch `main`, entrypoint `midterm_bikeshare_app.py`.
5. Test all tabs online and submit its live URL plus repository URL.

## Study notes and limitations
- Every row in source represents a ride, not a distinct person.
- Only ride-start time is used for temporal aggregation; local timestamps carry no explicit timezone.
- Station map/network count only rides with start and end station IDs. Dockless rides without station IDs still count in the time series.
- The network diagram combines directions to avoid clutter, whereas the underlying `daily_routes.csv` retains directions. Node positions are not geographic.
- A map location never represents the actual streets the cyclist traveled.
- There are no individual rider identities in the dataset.
- The main demo is one month: do not claim it proves seasonal patterns.

## Accessibility
Descriptive tab headings, color-safe scales, hover labels, visible legends, written findings and expandable accessible data tables accompany all charts.

## Data license
Review the Capital Bikeshare System Data page and linked Data License Agreement before redistributing aggregates or raw files.
