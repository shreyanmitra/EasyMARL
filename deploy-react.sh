#!/bin/bash

# EasyMARL React Deployment Script
# This script builds and deploys the React frontend to GitHub Pages

echo "🚀 EasyMARL React Deployment Script"
echo "==================================="

# Check if we're in the right directory
if [ ! -f "react-frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the EasyMARL root directory"
    exit 1
fi

# Navigate to React frontend directory
cd react-frontend

echo "📦 Installing dependencies..."
npm install

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo "🔨 Building React app for production..."
npm run build

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to build React app"
    exit 1
fi

echo "🌐 Deploying to GitHub Pages..."
npm run deploy

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo "✅ Successfully deployed to GitHub Pages!"
    echo "🔗 Your app should be available at: https://yourusername.github.io/EasyMARL"
    echo ""
    echo "📝 Next steps:"
    echo "1. Update the GitHub repository URL in package.json"
    echo "2. Configure your backend API URL in .env.production"
    echo "3. Set up backend server for full functionality"
    echo ""
    echo "📚 See react-deployment-guide.md for detailed instructions"
else
    echo "❌ Error: Failed to deploy to GitHub Pages"
    echo "🔧 Please check your GitHub repository settings and try again"
    exit 1
fi
