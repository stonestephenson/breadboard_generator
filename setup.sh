#!/bin/bash
# setup.sh — Initialize the breadboard generator project
# Run this once after cloning/downloading the project files.

set -e

echo "=== Breadboard Synthetic Data Generator V2 ==="
echo "=== Environment Setup ==="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found Python: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# Create directory structure
echo "Creating project directories..."
mkdir -p config/circuits
mkdir -p generator
mkdir -p reference
mkdir -p tests
mkdir -p output

# Create __init__.py if missing
touch generator/__init__.py

# Create .gitkeep files
touch output/.gitkeep

# Initialize git if not already a repo
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    
    # Create .gitignore
    cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
.pytest_cache/
output/images/
output/dataset_*/
*.egg-info/
.DS_Store
EOF
    
    git add -A
    git commit -m "Initial project setup: architecture docs, board spec, project structure"
    echo "Git repository initialized with initial commit."
else
    echo "Git repository already exists."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests (once implemented):"
echo "  pytest tests/ -v"
echo ""
echo "Next step: Open Claude Code in this directory and tell it:"
echo '  "Read CLAUDE.md and ARCHITECTURE.md. Then implement Phase 1."'
echo ""
