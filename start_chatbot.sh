#!/bin/bash
# Instructional material — CEU AI Engineering Class
# Usage: ./start_chatbot.sh chatbot/1_simple_chatbot.py

if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-chatbot-script>"
  echo "Example: $0 chatbot/1_simple_chatbot.py"
  exit 1
fi

echo "Starting chatbot: $1"
echo "Listening on http://0.0.0.0:10000"
echo "Press Ctrl+C to stop."
echo ""

chainlit run "$1" --port 10000 --host 0.0.0.0
