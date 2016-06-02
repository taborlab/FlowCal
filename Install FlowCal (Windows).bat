:: Not show commands
@echo off
:: Install FlowCal
python -m pip install -r requirements.txt --no-cache-dir
python setup.py install
:: Pause
set /p=Press [Enter] to finish...