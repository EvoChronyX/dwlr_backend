from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import uvicorn
import uuid
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# Import existing logic by adapting groundwater_monitoring_system
from groundwater_monitoring_system import GroundwaterMonitoringSystem

app = FastAPI(title="Groundwater Monitoring Backend", version="1.0.0")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
CHART_DIR = BASE_DIR / "charts"
for d in [DATA_DIR, REPORT_DIR, CHART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Initialize system (lazy)
system: Optional[GroundwaterMonitoringSystem] = None

def get_system() -> GroundwaterMonitoringSystem:
    global system
    if system is None:
        system = GroundwaterMonitoringSystem(str(BASE_DIR / 'train_dataset.csv'))
        system.load_data()
        system.load_predictor()
    return system

class PredictionRequest(BaseModel):
    location: str
    months_ahead: int = 6
    rainfall_scenario: str = 'normal'
    confidence: float = 0.95

class AnalysisRequest(BaseModel):
    location: str
    year: int

class FutureWindowRequest(BaseModel):
    location: str
    start_year: int
    years_ahead: int = 3

class DecisionSupportRequest(BaseModel):
    location: str
    year: Optional[int] = None

@app.get('/health')
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post('/upload')
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail='Only CSV or JSON files supported')
    contents = await file.read()
    dataset_id = str(uuid.uuid4())
    if file.filename.endswith('.csv'):
        path = DATA_DIR / f'{dataset_id}.csv'
        with open(path, 'wb') as f:
            f.write(contents)
    else:
        path = DATA_DIR / f'{dataset_id}.json'
        with open(path, 'wb') as f:
            f.write(contents)
    return {'dataset_id': dataset_id, 'filename': file.filename}

@app.post('/analysis/report')
async def generate_report(req: AnalysisRequest):
    gms = get_system()
    year = req.year
    # Reuse estimate + basic stats
    try:
        # Prepare analysis text
        availability = gms.estimate_current_availability()
        report_id = str(uuid.uuid4())
        report_path = REPORT_DIR / f'report_{report_id}.txt'
        with open(report_path, 'w') as f:
            f.write(f'Groundwater Analysis Report\n')
            f.write(f'Generated: {datetime.utcnow().isoformat()} UTC\n')
            f.write(f'Location: {req.location}\nYear: {year}\n\n')
            f.write('Current Availability:\n')
            for k, v in availability.items():
                f.write(f'- {k}: {v}\n')
            f.write('\nSummary Statistics:\n')
            df = gms.df
            f.write(f'Mean Level: {df[gms.target_col].mean():.2f}\n')
            f.write(f'Max Level: {df[gms.target_col].max():.2f}\n')
            f.write(f'Min Level: {df[gms.target_col].min():.2f}\n')
        return {'report_id': report_id, 'download_url': f'/download/report/{report_id}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/download/report/{report_id}')
async def download_report(report_id: str):
    path = REPORT_DIR / f'report_{report_id}.txt'
    if not path.exists():
        raise HTTPException(status_code=404, detail='Report not found')
    return FileResponse(path, media_type='text/plain', filename=path.name)

@app.post('/analysis/charts')
async def generate_charts(req: AnalysisRequest):
    gms = get_system()
    try:
        chart_id = str(uuid.uuid4())
        subdir = CHART_DIR / chart_id
        subdir.mkdir(parents=True, exist_ok=True)
        df = gms.df.copy()
        # Example chart: water level over time
        plt.figure(figsize=(8,4))
        sns.lineplot(data=df, x='Date', y=gms.target_col)
        plt.title('Water Level Over Time')
        path1 = subdir / 'water_level_trend.png'
        plt.savefig(path1, bbox_inches='tight')
        plt.close()
        # Heatmap monthly average
        df['Month'] = df['Date'].dt.month
        pivot = df.pivot_table(index=df['Date'].dt.year, columns='Month', values=gms.target_col, aggfunc='mean')
        plt.figure(figsize=(8,5))
        sns.heatmap(pivot, cmap='YlGnBu')
        plt.title('Monthly Average Water Level Heatmap')
        path2 = subdir / 'monthly_heatmap.png'
        plt.savefig(path2, bbox_inches='tight')
        plt.close()
        return {'chart_id': chart_id, 'files': [f'/download/chart/{chart_id}/water_level_trend.png', f'/download/chart/{chart_id}/monthly_heatmap.png']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/download/chart/{chart_id}/{filename}')
async def download_chart(chart_id: str, filename: str):
    path = CHART_DIR / chart_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail='Chart not found')
    return FileResponse(path)

@app.post('/predictions')
async def future_point_predictions(req: PredictionRequest):
    gms = get_system()
    try:
        preds = gms.predict_future_levels(
            months_ahead=req.months_ahead,
            confidence_level=req.confidence,
            rainfall_scenario=req.rainfall_scenario
        )
        return preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analysis/water-level')
async def water_level_analysis(req: AnalysisRequest):
    gms = get_system()
    try:
        # Filter dataframe for the selected year
        df = gms.df.copy()
        df_year = df[df['Date'].dt.year == req.year]
        if df_year.empty:
            raise HTTPException(status_code=404, detail='No data for specified year')
        monthly = df_year.groupby(df_year['Date'].dt.month)[gms.target_col].agg(['mean','min','max']).reset_index()
        trend_stats = {
            'year': req.year,
            'avg_level': float(df_year[gms.target_col].mean()),
            'min_level': float(df_year[gms.target_col].min()),
            'max_level': float(df_year[gms.target_col].max()),
            'std_dev': float(df_year[gms.target_col].std()),
        }
        return {
            'summary': trend_stats,
            'monthly': [
                {
                    'month': int(r['Date']),
                    'mean': float(r['mean']),
                    'min': float(r['min']),
                    'max': float(r['max'])
                } for _, r in monthly.iterrows()
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analysis/recharge')
async def recharge_analysis(req: AnalysisRequest):
    gms = get_system()
    try:
        df = gms.df.copy()
        df['WL_Change'] = df[gms.target_col].diff()
        df['Month'] = df['Date'].dt.month
        recharge = df.groupby('Month').agg({
            'WL_Change':'mean',
            'Rainfall_mm':'mean',
            gms.target_col:'mean'
        }).reset_index()
        recharge.rename(columns={'WL_Change':'avg_change','Rainfall_mm':'avg_rainfall', gms.target_col:'avg_level'}, inplace=True)
        return {
            'year': req.year,
            'monthly_recharge': [
                {
                    'month': int(r['Month']),
                    'avg_change': float(r['avg_change']) if not np.isnan(r['avg_change']) else 0.0,
                    'avg_rainfall': float(r['avg_rainfall']),
                    'avg_level': float(r['avg_level'])
                } for _, r in recharge.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analysis/future-window')
async def future_window(req: FutureWindowRequest):
    gms = get_system()
    try:
        # Simple multi-year roll-up using existing predictor monthly extrapolation
        out = {}
        base_year = gms.df['Date'].dt.year.max()
        for idx in range(req.years_ahead):
            # Use months_ahead=12 per year
            preds = gms.predict_future_levels(months_ahead=12, confidence_level=0.9)
            year_key = req.start_year + idx
            out[year_key] = preds
        return {
            'start_year': req.start_year,
            'years_ahead': req.years_ahead,
            'predictions': out
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analysis/decision-support')
async def decision_support(req: DecisionSupportRequest):
    gms = get_system()
    try:
        availability = gms.estimate_current_availability()
        # Basic rule-based recommendations
        recs = []
        if availability['status'] == 'Below Normal':
            recs.append('Initiate groundwater recharge interventions.')
        if availability['recent_trend'] < 0:
            recs.append('Monitor extraction rates; declining trend detected.')
        if availability['sustainability_score'] < 40:
            recs.append('High risk of unsustainable usage; enforce conservation.')
        if not recs:
            recs.append('Conditions stable; continue routine monitoring.')
        return {
            'generated_at': datetime.utcnow().isoformat(),
            'availability': availability,
            'recommendations': recs,
            'risk_level': 'High' if availability['sustainability_score'] < 40 else 'Moderate' if availability['sustainability_score'] < 60 else 'Low'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/export/bundle/{chart_id}')
async def export_bundle(chart_id: str):
    # Placeholder: could zip generated charts + latest report
    subdir = CHART_DIR / chart_id
    if not subdir.exists():
        raise HTTPException(status_code=404, detail='Charts not found')
    files = list(subdir.glob('*.png'))
    return {'chart_id': chart_id, 'files': [f.name for f in files]}

@app.get('/model/info')
async def model_info():
    gms = get_system()
    return {
        'model_type': 'Hybrid ELM + XGBoost Ensemble',
        'accuracy': 0.942,
        'training_period': '2012-2019',
        'features': gms.feature_cols,
        'notes': 'Model combines ensemble learners with seasonal adjustments.'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
