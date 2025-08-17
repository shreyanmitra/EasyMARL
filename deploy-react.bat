@echo off
REM EasyMARL React Deployment Script for Windows
REM This script builds and deploys the React frontend to GitHub Pages

echo 🚀 EasyMARL React Deployment Script
echo ===================================

REM Check if we're in the right directory
if not exist "react-frontend\package.json" (
    echo ❌ Error: Please run this script from the EasyMARL root directory
    pause
    exit /b 1
)

REM Navigate to React frontend directory
cd react-frontend

echo 📦 Installing dependencies...
npm install

REM Check if installation was successful
if errorlevel 1 (
    echo ❌ Error: Failed to install dependencies
    pause
    exit /b 1
)

echo 🔨 Building React app for production...
npm run build

REM Check if build was successful
if errorlevel 1 (
    echo ❌ Error: Failed to build React app
    pause
    exit /b 1
)

echo 🌐 Deploying to GitHub Pages...
npm run deploy

REM Check if deployment was successful
if errorlevel 0 (
    echo ✅ Successfully deployed to GitHub Pages!
    echo 🔗 Your app should be available at: https://yourusername.github.io/EasyMARL
    echo.
    echo 📝 Next steps:
    echo 1. Update the GitHub repository URL in package.json
    echo 2. Configure your backend API URL in .env.production
    echo 3. Set up backend server for full functionality
    echo.
    echo 📚 See react-deployment-guide.md for detailed instructions
) else (
    echo ❌ Error: Failed to deploy to GitHub Pages
    echo 🔧 Please check your GitHub repository settings and try again
)

pause
