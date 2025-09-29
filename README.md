# Groundwater Monitoring Backend

FastAPI service providing ML-driven groundwater analytics, chart generation, and report export derived from existing `groundwater_monitoring_system.py` logic.

## Endpoints
- `GET /health` – service status
- `POST /upload` – upload CSV/JSON dataset
- `POST /analysis/report` – text report generation (returns download URL)
- `POST /analysis/charts` – generates and stores PNG charts (returns file URLs)
- `POST /analysis/water-level` – yearly water level summary + monthly stats
- `POST /analysis/recharge` – recharge pattern metrics
- `POST /predictions` – short‑term (months) predictions with confidence
- `POST /analysis/future-window` – multi‑year forward window aggregation
- `POST /analysis/decision-support` – availability, risk, recommendations
- `GET /model/info` – static model metadata
- `GET /download/report/{id}` – download report
- `GET /download/chart/{chart_id}/{filename}` – download generated chart
- `GET /export/bundle/{chart_id}` – list chart bundle files

## Local Development
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open: http://127.0.0.1:8000/docs for interactive API docs.

## Deployment (Render example)
1. Create new Web Service
2. Environment: Python
3. Start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
4. Add a persistent disk if you need stored charts/reports (optional)
5. Set `PYTHON_VERSION` in env vars if needed

## TODO (Next Iterations)
- Add PDF report variant
- Zip bundle export
- Authentication / API key
- Model registry & dynamic model reload
- Caching layer for repeated requests
