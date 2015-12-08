:: Not show commands
@echo off
:: Install FlowCal
pip install -r requirements.txt
python setup.py install
:: Pause
set /p=Press [Enter] to finish...