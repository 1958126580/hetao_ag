@echo off
chcp 65001
echo ========================================================
echo   hetao_ag GitHub Deployment Script
echo ========================================================
echo.
echo Due to GitHub security policies, automated password authentication
echo is no longer supported for Git operations. You must use a 
echo Personal Access Token (PAT) or SSH key, or push manually.
echo.
echo [Step 1] Please go to https://github.com/new and create a 
echo          repository named "hetao_ag".
echo          (Do not initialize with README, .gitignore, or License)
echo.
echo [Step 2] Enter your GitHub username below.
set /p COMP_USER=GitHub Username (e.g., WangZhihua): 

if "%COMP_USER%"=="" goto error

echo.
echo Setting remote to: https://github.com/%COMP_USER%/hetao_ag.git
git remote remove origin 2>nul
git remote add origin https://github.com/%COMP_USER%/hetao_ag.git

echo.
echo [Step 3] Pushing to GitHub...
echo You will be asked to sign in. A browser window may open,
echo or you may need to paste a Personal Access Token (PAT).
echo.
pause

git branch -M main
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Push failed. 
    echo Possible reasons:
    echo  - Repository does not exist (did you create it?)
    echo  - Authentication failed (password auth is disabled, use Token)
    echo  - Network issue
    goto end
)

echo.
echo [SUCCESS] Successfully pushed to GitHub!
echo Repository: https://github.com/%COMP_USER%/hetao_ag
goto end

:error
echo Error: Username is required.

:end
pause
