@echo off
chcp 65001
echo ========================================================
echo   hetao_ag GitHub CLI Upload Script
echo ========================================================
echo.

echo [Step 1] Checking GitHub CLI (gh)...
where gh >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 'gh' command not found!
    echo.
    echo Since you just installed GitHub CLI, you MUST restart 
    echo your terminal or VS Code to update the system PATH.
    echo.
    echo Please close this terminal, open a new one, and run this script again.
    pause
    exit /b
)

echo [Step 2] Authenticating with GitHub...
echo You will be asked to choose "GitHub.com" and then "Login with a web browser".
gh auth login

echo.
echo [Step 3] Creating repository 'hetao_ag'...
gh repo create hetao_ag --public --source=. --remote=origin --push

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Repo creation failed. It might already exist.
    echo Attempting to push mainly...
    git remote add origin https://github.com/%USERNAME%/hetao_ag.git 2>nul
    git branch -M main
    git push -u origin main
)

echo.
echo [COMPLETE] Deployment finished.
pause
