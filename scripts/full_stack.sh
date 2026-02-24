#!/usr/bin/env bash
#
# Orion Full Stack Launcher
# Starts backend (FastAPI) and frontend (SvelteKit) simultaneously.
#
# Usage:
#   ./full_stack.sh           # Auto-detect or prompt
#   ./full_stack.sh local     # Local mode (opens browser)
#   ./full_stack.sh vm        # VM mode (binds to 0.0.0.0)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# Track PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    info "Shutting down Orion stack..."
    
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
        success "Backend stopped"
    fi
    
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
        success "Frontend stopped"
    fi
    
    # Kill any processes on the ports
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 5173/tcp 2>/dev/null || true
    
    success "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Detect environment
detect_environment() {
    # Check for SSH connection
    if [ -n "$SSH_CONNECTION" ] || [ -n "$SSH_CLIENT" ]; then
        echo "vm"
        return
    fi
    
    # Check for display
    if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
        echo "vm"
        return
    fi
    
    # Check GCP metadata
    if curl -s -f -m 1 "http://metadata.google.internal/computeMetadata/v1/" -H "Metadata-Flavor: Google" >/dev/null 2>&1; then
        echo "vm"
        return
    fi
    
    echo "local"
}

# Get run mode
get_run_mode() {
    local mode="${1:-}"
    
    if [ -n "$mode" ]; then
        echo "$mode"
        return
    fi
    
    local detected
    detected=$(detect_environment)
    
    if [ "$detected" = "vm" ]; then
        info "Detected VM/remote environment"
        echo "vm"
        return
    fi
    
    echo ""
    echo "Select run mode:"
    echo "  [1] Local  - Opens browser, localhost access"
    echo "  [2] VM     - Headless, network access (0.0.0.0)"
    echo ""
    read -rp "Enter choice (1 or 2): " choice
    
    case "$choice" in
        1) echo "local" ;;
        2) echo "vm" ;;
        *) 
            warn "Invalid choice, defaulting to local"
            echo "local"
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Python
    if command -v python3 &>/dev/null; then
        success "Python: $(python3 --version)"
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        success "Python: $(python --version)"
        PYTHON_CMD="python"
    else
        error "Python not found. Please install Python 3.10+"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &>/dev/null; then
        success "Node.js: $(node --version)"
    else
        error "Node.js not found. Please install Node.js 18+"
        exit 1
    fi
    
    # Check/create venv
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        warn "Virtual environment not found"
        info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$PROJECT_ROOT/.venv"
    fi
    
    # Check/install node_modules
    if [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
        warn "node_modules not found"
        info "Installing frontend dependencies..."
        (cd "$PROJECT_ROOT/frontend" && npm install)
    fi
    
    success "Prerequisites OK"
}

# Start the stack
start_stack() {
    local run_mode="$1"
    
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}       ORION FULL STACK LAUNCHER       ${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""
    
    check_prerequisites
    
    info "Starting in $run_mode mode..."
    echo ""
    
    # Determine host binding
    local backend_host frontend_host
    if [ "$run_mode" = "vm" ]; then
        backend_host="0.0.0.0"
        frontend_host="0.0.0.0"
    else
        backend_host="127.0.0.1"
        frontend_host="localhost"
    fi
    
    # Activate venv
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
    
    # Start Backend
    info "Starting backend on $backend_host:8000..."
    (cd "$PROJECT_ROOT" && uvicorn backend.app:app --host "$backend_host" --port 8000 --reload) &
    BACKEND_PID=$!
    
    sleep 2
    
    # Start Frontend
    info "Starting frontend on $frontend_host:5173..."
    if [ "$run_mode" = "vm" ]; then
        (cd "$PROJECT_ROOT/frontend" && npm run dev -- --host 0.0.0.0) &
    else
        (cd "$PROJECT_ROOT/frontend" && npm run dev) &
    fi
    FRONTEND_PID=$!
    
    sleep 3
    
    # Display access info
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}         ORION IS RUNNING              ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    if [ "$run_mode" = "vm" ]; then
        local external_ip
        external_ip=$(curl -s -f -m 5 http://checkip.amazonaws.com 2>/dev/null || hostname -I | awk '{print $1}')
        echo -e "  Frontend:  http://$external_ip:5173"
        echo -e "  Backend:   http://$external_ip:8000"
        echo -e "  API Docs:  http://$external_ip:8000/docs"
    else
        echo -e "  Frontend:  http://localhost:5173"
        echo -e "  Backend:   http://localhost:8000"
        echo -e "  API Docs:  http://localhost:8000/docs"
    fi
    
    echo ""
    echo -e "${YELLOW}  Press Ctrl+C to stop all services${NC}"
    echo ""
    
    # Open browser in local mode
    if [ "$run_mode" = "local" ]; then
        if command -v xdg-open &>/dev/null; then
            xdg-open "http://localhost:5173" 2>/dev/null || true
        elif command -v open &>/dev/null; then
            open "http://localhost:5173" 2>/dev/null || true
        fi
    fi
    
    # Wait for processes
    wait
}

# Main
main() {
    local run_mode
    run_mode=$(get_run_mode "$1")
    start_stack "$run_mode"
}

main "$@"
