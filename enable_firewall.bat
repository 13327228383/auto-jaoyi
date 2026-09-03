@echo off
REM Enable Windows Firewall (run as Administrator: right-click -> Run as administrator)
REM Real-machine check showed all 3 firewall profiles are OFF; a money-trading PC must have it ON.
REM If still OFF after running, Group Policy (GPO) is forcing it; fix in local/domain GPO.
echo Enabling Windows Firewall (requires admin)...
netsh advfirewall set allprofiles state on
echo.
echo Current status:
netsh advfirewall show allprofiles
echo.
echo Done. "State = ON" means success; if still OFF, check Group Policy.
pause
