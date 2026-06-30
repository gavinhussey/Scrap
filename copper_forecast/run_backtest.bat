@echo off
cd /d "C:\dev\scrap - ghussey\Scrap\copper_forecast"
set PYTHONUNBUFFERED=1
set TF_ENABLE_ONEDNN_OPTS=0
echo [%date% %time%] Backtest started >> backtest_full_log.txt
".venv\Scripts\python.exe" -u copper_lstm.py --retrain-every 1 --epochs 1000 >> backtest_full_log.txt 2>&1
echo [%date% %time%] Backtest finished >> backtest_full_log.txt
