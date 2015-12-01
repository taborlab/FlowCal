:: Not show commands
@echo off
:: Install fc
pip install -r requirements.txt
python setup.py install
:: Pause
set /p=Press [Enter] to finish...