:: Not show commands
@echo off
:: Run FlowCal
python -m FlowCal.excel_ui -v -p -H
:: Pause
set /p=Press [Enter] to finish...
