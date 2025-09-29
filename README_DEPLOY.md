# Deployment Guide (Render)

This guide explains how to deploy the FastAPI groundwater monitoring backend to Render.

## 1. Repository Structure
```
/ (repo root)
  groundwater_monitoring_system.py
  train_dataset.csv
  /backend
    main.py
    requirements.txt
    render.yaml
    Dockerfile (optional for local) 
    README.md
```

## 2. One-Click Render Setup
If your repo root contains `backend/render.yaml`, Render can auto-detect it.

Steps:
1. Push all changes to GitHub.
2. In Render dashboard: New + Web Service.
3. Select the repository.
4. If using render.yaml: Render will create the service automatically.
5. Otherwise set parameters manually:
   - Environment: Python
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## 3. Persistent Data (Optional)
The service definition in `render.yaml` provisions a 1 GB disk mounted at `/opt/render/project/src/backend/data` for uploaded datasets. Adjust size as needed.

## 4. Environment Variables
`PYTHON_VERSION` is set in render.yaml. Add others if you introduce secrets or toggles.

## 5. Testing Locally
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Open: http://127.0.0.1:8000/docs

## 6. Health Check
Render hits `/health`. Ensure it returns status 200. Already implemented.

## 7. Logs & Monitoring
Use Render dashboard -> Logs to monitor startup, errors, and request traces.

## 8. Updating the Service
Push commits to the tracked branch. AutoDeploy is enabled; service rebuilds automatically.

## 9. Scaling Considerations
- Free plan sleeps when idle; predictions may have cold start latency.
- For higher concurrency: upgrade plan, enable autoscaling, add caching layer (future).

## 10. Next Enhancements
- Add `/export/bundle` to generate ZIP of charts + report.
- Add authentication via API key header.
- Add PDF generation endpoint.
- Implement model versioning & hot reload.

## 11. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Module not found | Wrong working dir | Ensure start command uses `backend.main:app` |
| 500 errors on prediction | Missing dataset | Upload CSV or ensure `train_dataset.csv` committed |
| Slow cold start | Free instance sleeping | Ping `/health` periodically |

## 12. Example cURL Calls
```bash
curl -X GET https://<your-service>.onrender.com/health
curl -X POST https://<your-service>.onrender.com/predictions -H "Content-Type: application/json" -d "{\"location\":\"siteA\",\"months_ahead\":3}"
```

---
Happy deploying!
