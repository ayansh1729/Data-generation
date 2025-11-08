#!/bin/bash
# Quick Setup Script for Data Generation Project
# Run this script to set up everything automatically

echo "🚀 Setting up Data Generation Project..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found!"
    echo "   Please run this script from the project root directory"
    exit 1
fi

echo "✅ Found project root"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install project as package
echo ""
echo "📦 Installing project as editable package..."
pip install -e .

# Install Jupyter
echo ""
echo "📓 Installing Jupyter..."
pip install jupyter ipywidgets ipykernel

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Activate the environment:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "      venv\\Scripts\\activate"
else
    echo "      source venv/bin/activate"
fi
echo "   2. Start Jupyter:"
echo "      jupyter notebook"
echo "   3. Open: notebooks/01_getting_started.ipynb"
echo ""
echo "🎉 Happy generating!"

