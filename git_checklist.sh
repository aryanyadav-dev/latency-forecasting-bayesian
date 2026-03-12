#!/bin/bash
# Script to prepare repository for safe GitHub push
# Removes confidential data, caches, and temporary files

set -e

echo "=========================================="
echo "Preparing Repository for GitHub Push"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if directory/file exists and remove
safe_remove() {
    if [ -e "$1" ]; then
        echo -e "${YELLOW}Removing:${NC} $1"
        rm -rf "$1"
    fi
}

# 1. Verify .kiro directory won't be pushed (CONFIDENTIAL - keep locally)
echo "1. Verifying .kiro directory protection..."
if [ -d ".kiro" ]; then
    echo -e "${YELLOW}Info:${NC} .kiro directory exists locally (this is correct)"
    if grep -q "^\.kiro/" .gitignore 2>/dev/null; then
        echo -e "${GREEN}✓${NC} .kiro/ is in .gitignore - will NOT be pushed to GitHub"
    else
        echo -e "${RED}✗${NC} WARNING: .kiro/ NOT in .gitignore!"
        echo "Adding .kiro/ to .gitignore..."
        echo "" >> .gitignore
        echo "# Kiro directory (CONFIDENTIAL - DO NOT PUSH)" >> .gitignore
        echo ".kiro/" >> .gitignore
        echo -e "${GREEN}✓${NC} Added .kiro/ to .gitignore"
    fi
else
    echo -e "${YELLOW}Info:${NC} .kiro directory does not exist (optional)"
fi
echo ""

# 2. Clear all caches
echo "2. Clearing caches..."
safe_remove ".cache"
safe_remove "data/cache"
safe_remove "data/.cache"
safe_remove ".pytest_cache"
safe_remove ".mypy_cache"
safe_remove "__pycache__"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓${NC} Caches cleared"
echo ""

# 3. Remove logs and results
echo "3. Removing logs and results..."
safe_remove "experiments/logs"
safe_remove "experiments/results"
find . -name "*.log" -delete 2>/dev/null || true
find . -name "*.csv" -delete 2>/dev/null || true
find . -name "events.out.tfevents.*" -delete 2>/dev/null || true
echo -e "${GREEN}✓${NC} Logs and results removed"
echo ""

# 4. Remove checkpoints (keep directory structure)
echo "4. Removing checkpoint files (keeping directory)..."
if [ -d "checkpoints" ]; then
    find checkpoints/ -name "*.pt" -delete 2>/dev/null || true
    find checkpoints/ -name "*.pth" -delete 2>/dev/null || true
    find checkpoints/ -name "*.ckpt" -delete 2>/dev/null || true
    # Ensure .gitkeep exists
    touch checkpoints/.gitkeep
fi
echo -e "${GREEN}✓${NC} Checkpoint files removed (directory preserved)"
echo ""

# 5. Remove temporary files
echo "5. Removing temporary files..."
safe_remove "tmp"
safe_remove "temp"
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.swp" -delete 2>/dev/null || true
find . -name "*.swo" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
echo -e "${GREEN}✓${NC} Temporary files removed"
echo ""

# 6. Remove data files (keep structure)
echo "6. Cleaning data directory..."
if [ -d "data" ]; then
    find data/ -name "*.txt" -delete 2>/dev/null || true
    find data/ -name "*.json" -delete 2>/dev/null || true
    find data/ -name "*.parquet" -delete 2>/dev/null || true
    find data/ -name "*.hdf5" -delete 2>/dev/null || true
    find data/ -name "*.h5" -delete 2>/dev/null || true
fi
echo -e "${GREEN}✓${NC} Data files cleaned"
echo ""

# 7. Remove virtual environment
echo "7. Checking virtual environment..."
if [ -d "venv" ] || [ -d "env" ] || [ -d ".venv" ]; then
    echo -e "${YELLOW}Warning:${NC} Virtual environment detected"
    echo "Virtual environments should not be pushed to GitHub"
    echo "They are already in .gitignore"
fi
echo ""

# 8. Verify .gitignore
echo "8. Verifying .gitignore..."
if [ -f ".gitignore" ]; then
    if grep -q ".kiro/" ".gitignore"; then
        echo -e "${GREEN}✓${NC} .kiro/ is in .gitignore"
    else
        echo -e "${RED}✗${NC} .kiro/ NOT in .gitignore - ADDING IT"
        echo "" >> .gitignore
        echo "# Kiro directory (CONFIDENTIAL - DO NOT PUSH)" >> .gitignore
        echo ".kiro/" >> .gitignore
    fi
else
    echo -e "${RED}✗${NC} .gitignore not found!"
fi
echo ""

# 9. Check for sensitive files
echo "9. Checking for sensitive files..."
SENSITIVE_FOUND=0

# Check for API keys or tokens
if grep -r "api_key\|API_KEY\|token\|TOKEN\|password\|PASSWORD" --include="*.py" --include="*.yaml" --include="*.yml" . 2>/dev/null | grep -v "# " | grep -v "test" | head -5; then
    echo -e "${RED}⚠️  Potential sensitive data found in code!${NC}"
    echo "Please review and remove before pushing"
    SENSITIVE_FOUND=1
fi

if [ $SENSITIVE_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No obvious sensitive data found"
fi
echo ""

# 10. Git status check
echo "10. Checking git status..."
if [ -d ".git" ]; then
    echo "Files to be committed:"
    git status --short | head -20
    echo ""
    echo "Total files changed:"
    git status --short | wc -l
else
    echo -e "${YELLOW}Warning:${NC} Not a git repository"
    echo "Run 'git init' to initialize"
fi
echo ""

# Summary
echo "=========================================="
echo "Preparation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Review changes:"
echo "   ${GREEN}git status${NC}"
echo ""
echo "2. Add files:"
echo "   ${GREEN}git add .${NC}"
echo ""
echo "3. Commit:"
echo "   ${GREEN}git commit -m \"Initial commit: Latent Forecasting Network implementation\"${NC}"
echo ""
echo "4. Add remote (if not already added):"
echo "   ${GREEN}git remote add origin https://github.com/your-username/latent-forecasting-network.git${NC}"
echo ""
echo "5. Push to GitHub:"
echo "   ${GREEN}git push -u origin main${NC}"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT:${NC} Review git status before pushing!"
echo "Ensure no confidential data is included."
echo ""
