#!/bin/bash
# Production startup script for 5G Backend

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Set default values
export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_DEBUG=${FLASK_DEBUG:-false}
export FLASK_HOST=${FLASK_HOST:-0.0.0.0}
export PORT=${PORT:-5001}

echo "🚀 Starting 5G Backend on port $PORT..."
echo "📍 Environment: $FLASK_ENV"
echo "🌐 Host: $FLASK_HOST"

# Start with Gunicorn (production WSGI server)
gunicorn --bind $FLASK_HOST:$PORT \
         --workers 2 \
         --worker-class sync \
         --timeout 300 \
         --access-logfile - \
         --error-logfile - \
         --log-level info \
         wsgi:app
