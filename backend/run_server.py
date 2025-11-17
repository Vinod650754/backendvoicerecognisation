#!/usr/bin/env python
import uvicorn
import time

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
    # Keep the process alive
    while True:
        time.sleep(1)
