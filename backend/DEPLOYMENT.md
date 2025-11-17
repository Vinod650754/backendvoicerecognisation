# Backend Deployment Guide

## Architecture Overview

The backend is organized into a modular, production-ready architecture:

```
backend/
├── app/                          # Main application package
│   ├── __init__.py              # FastAPI application factory
│   ├── config/                  # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py          # Environment variables & settings
│   │   └── logger.py            # Logging configuration
│   ├── models/                  # Data models and schemas
│   │   ├── __init__.py
│   │   ├── neural_models.py     # PyTorch neural network models
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/                # Business logic layer
│   │   ├── __init__.py
│   │   └── inference_service.py # Audio processing & model inference
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   └── audio_processor.py   # Audio preprocessing utilities
│   └── routes/                  # API endpoint routes
│       ├── __init__.py
│       ├── health.py            # Health & status endpoints
│       ├── samples.py           # Audio sample collection
│       ├── detection.py         # Wakeword detection
│       └── verification.py      # Command detection & voice verification
├── data/                        # Data storage (created at runtime)
│   ├── wakeword/               # Wakeword training samples
│   ├── command/                # Command training samples
│   ├── open_door/              # Open door command samples
│   └── close_door/             # Close door command samples
├── Dockerfile                   # Docker containerization
├── docker-compose.yml           # Docker Compose orchestration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── START_BACKEND.bat            # Windows startup script
├── start_backend.sh             # Linux/Mac startup script
└── README.md                    # This file
```

## Quick Start

### Local Development (Windows)

1. **Activate Virtual Environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run the Backend**
   ```powershell
   .\backend\START_BACKEND.bat
   ```

Or directly:
   ```powershell
   cd backend
   python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

3. **Access the API**
   - API Base: http://127.0.0.1:8000
   - Interactive Docs: http://127.0.0.1:8000/docs
   - Alternative Docs: http://127.0.0.1:8000/redoc

### Local Development (Linux/Mac)

```bash
chmod +x backend/start_backend.sh
./backend/start_backend.sh
```

### Docker Deployment

**Build and Run with Docker:**
```bash
cd backend
docker-compose up --build
```

**Build Image Manually:**
```bash
docker build -t voice-backend:latest .
docker run -p 8000:8000 -v $(pwd)/data:/app/data voice-backend:latest
```

## API Endpoints

### Health & Status

```
GET /health
GET /
```

### Audio Sample Collection

```
POST /collect_sample
Content-Type: multipart/form-data

Parameters:
  - file: Audio WAV file
  - label: string (wakeword, command, open_door, close_door)

Response:
{
  "success": true,
  "message": "Sample saved as sample_1234567890.wav",
  "path": "/path/to/backend/data/wakeword/sample_1234567890.wav",
  "label": "wakeword",
  "timestamp": 1234567890
}
```

### Wakeword Detection

```
POST /detect_wakeword
Content-Type: multipart/form-data

Parameters:
  - file: Audio WAV file

Response:
{
  "wakeword_detected": true,
  "confidence": 0.87,
  "processing_time_ms": 245.3
}
```

### Command Detection

```
POST /detect_command
Content-Type: multipart/form-data

Parameters:
  - file: Audio WAV file

Response:
{
  "intent": "open",
  "intent_idx": 2,
  "confidence": 0.92,
  "processing_time_ms": 198.7
}
```

### Voice Verification

```
POST /verify_voice
Content-Type: multipart/form-data

Parameters:
  - file: Audio WAV file
  - user_id: string (optional, default: "HomeOwner")

Response:
{
  "verified": true,
  "confidence": 0.78,
  "threshold": 0.7,
  "processing_time_ms": 312.4
}
```

## Testing with Postman

### 1. Health Check
- **Method:** GET
- **URL:** http://127.0.0.1:8000/health
- **Expected Response:** 200 OK

### 2. Upload Sample
- **Method:** POST
- **URL:** http://127.0.0.1:8000/collect_sample
- **Body (form-data):**
  - `file`: Select an audio WAV file
  - `label`: wakeword (or command, open_door, close_door)
- **Expected Response:** 200 OK with file path

### 3. Detect Wakeword
- **Method:** POST
- **URL:** http://127.0.0.1:8000/detect_wakeword
- **Body (form-data):**
  - `file`: Select an audio WAV file
- **Expected Response:** 200 OK with detection result

### 4. Detect Command
- **Method:** POST
- **URL:** http://127.0.0.1:8000/detect_command
- **Body (form-data):**
  - `file`: Select an audio WAV file
- **Expected Response:** 200 OK with command intent

### 5. Verify Voice
- **Method:** POST
- **URL:** http://127.0.0.1:8000/verify_voice
- **Body (form-data):**
  - `file`: Select an audio WAV file
- **Expected Response:** 200 OK with verification result

## Testing with Python

See `test_client.py` for comprehensive Python test client.

## Configuration

Edit `.env` file or set environment variables:

```bash
HOST=127.0.0.1
PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO
WAKEWORD_CONFIDENCE_THRESHOLD=0.7
VOICE_VERIFICATION_THRESHOLD=0.7
```

## Logging

Logs are written to:
- Console (stdout)
- `backend.log` file

Set `LOG_LEVEL` environment variable:
- DEBUG: Verbose output
- INFO: General information
- WARNING: Warnings only
- ERROR: Errors only

## Data Storage

Audio samples are automatically organized by label:
- `data/wakeword/` - Wakeword training samples
- `data/command/` - Generic command samples
- `data/open_door/` - Open door samples
- `data/close_door/` - Close door samples

## Performance Notes

- Audio preprocessing: 50-300ms per file
- Model inference: 100-500ms per file
- Mel spectrogram computation: Most of the processing time
- Models run on CPU (for on-device compatibility)
- Typical end-to-end latency: 200-800ms

## Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Import Errors
Ensure you're running from project root with virtual environment activated:
```bash
cd /path/to/home_app
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac
```

### Audio Processing Errors
- Check audio is WAV format
- Audio duration should be 0.5-30 seconds
- Sample rate: 8000-48000 Hz recommended

## Production Deployment

### Using Docker

1. **Build image:**
   ```bash
   docker build -t voice-backend:1.0.0 .
   ```

2. **Run container:**
   ```bash
   docker run -p 8000:8000 \
     -v /data/voice-samples:/app/data \
     -e ENVIRONMENT=production \
     -e LOG_LEVEL=INFO \
     voice-backend:1.0.0
   ```

3. **Using docker-compose:**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.voiceauth.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Security Notes

- Enable HTTPS in production (add SSL certificates)
- Use environment variables for sensitive config
- Limit CORS origins to known clients only
- Add authentication for production (JWT, API keys)
- Set proper file upload size limits
- Validate all input data
- Run with least privilege user

## Dependencies

See `requirements.txt`:
- FastAPI: Web framework
- Uvicorn: ASGI server
- PyTorch: Neural network inference
- torchaudio: Audio processing
- librosa: Audio feature extraction
- soundfile: WAV file I/O
- Pydantic: Data validation

## Contributing

When adding new features:
1. Add model definitions to `app/models/`
2. Add business logic to `app/services/`
3. Add API routes to `app/routes/`
4. Update schemas in `app/models/schemas.py`
5. Test with Postman or `test_client.py`
6. Update this README

## Support

For issues or questions:
1. Check logs in console and `backend.log`
2. Review API documentation at http://127.0.0.1:8000/docs
3. Verify data files in `backend/data/` directory
4. Check audio files are valid WAV format
