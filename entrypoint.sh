#!/bin/sh
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 &
nginx -g 'daemon off;'
