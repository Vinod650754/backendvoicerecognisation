# Backend Architecture Documentation

## Overview

The backend has been restructured from a monolithic `app.py` into a modular, production-ready architecture following FastAPI best practices and clean architecture principles.

## New Structure

```
backend/
├── app/                           # Main application package
│   ├── __init__.py               # FastAPI app factory
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py          # Environment variables & app settings
│   │   └── logger.py            # Centralized logging setup
│   ├── models/                   # Domain models & schemas
│   │   ├── __init__.py
│   │   ├── neural_models.py     # PyTorch model definitions
│   │   │   - WakeWordModel       (Binary classifier)
│   │   │   - CommandModel        (Multi-class classifier)
│   │   │   - VoiceBiometricModel (Speaker embeddings)
│   │   └── schemas.py           # Pydantic request/response schemas
│   │       - HealthResponse
│   │       - SampleUploadResponse
│   │       - WakewordDetectionResponse
│   │       - CommandDetectionResponse
│   │       - VoiceVerificationResponse
│   │       - ErrorResponse
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   └── inference_service.py # Model inference & audio processing
│   │       - AudioProcessingService
│   │       - WakewordDetectionService
│   │       - CommandDetectionService
│   │       - VoiceVerificationService
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── audio_processor.py   # Audio preprocessing helpers
│   │       - preprocess_audio_bytes()
│   │       - validate_audio_file()
│   └── routes/                   # API endpoint routes
│       ├── __init__.py
│       ├── health.py            # Health check endpoints
│       │   - GET /health
│       │   - GET /
│       ├── samples.py           # Sample collection endpoints
│       │   - POST /collect_sample
│       ├── detection.py         # Wakeword detection endpoint
│       │   - POST /detect_wakeword
│       └── verification.py      # Command & voice verification endpoints
│           - POST /detect_command
│           - POST /verify_voice
├── data/                         # Audio data storage (auto-created)
│   ├── wakeword/                # Wakeword training samples
│   ├── command/                 # Generic command samples
│   ├── open_door/               # Open door command samples
│   └── close_door/              # Close door command samples
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── START_BACKEND.bat             # Windows startup script
├── start_backend.sh              # Linux/Mac startup script
├── test_client_comprehensive.py  # Comprehensive test suite
├── DEPLOYMENT.md                 # Detailed deployment guide
└── README.md                     # Project overview
```

## Key Improvements

### 1. **Separation of Concerns**
- **Routes**: HTTP layer handles requests/responses
- **Services**: Business logic isolated for reusability
- **Models**: Data validation and neural network definitions
- **Utils**: Pure functions for common tasks
- **Config**: Centralized configuration management

### 2. **Configuration Management**
- Environment variables via `.env` file
- Settings centralized in `app/config/settings.py`
- Easy to switch between dev/staging/prod environments
- All configurable thresholds in one place

### 3. **Logging**
- Centralized logger setup in `app/config/logger.py`
- Logs to both console and `backend.log` file
- Configurable log levels
- Structured logging with timestamps

### 4. **Error Handling**
- Pydantic schemas for request/response validation
- Consistent error response format
- Detailed error messages for debugging
- Graceful error handling in all services

### 5. **Model Management**
- Neural models defined in `app/models/neural_models.py`
- Easy to add new models without changing routes
- Models instantiated as singletons in services
- Support for model versioning

### 6. **Testing**
- Comprehensive test client included
- Easy to add unit tests per module
- Integration tests can test full pipeline

## Data Flow

```
HTTP Request (Client/Frontend)
    ↓
Route Handler (app/routes/*.py)
    ↓
Schema Validation (Pydantic)
    ↓
Service Layer (app/services/*.py)
    ├→ Audio Processing (app/utils/audio_processor.py)
    ├→ Model Inference (app/models/neural_models.py)
    └→ Business Logic
    ↓
Schema Response (Pydantic)
    ↓
HTTP Response (JSON)
```

## Adding New Features

### Adding a New Endpoint

1. **Create route file** `app/routes/my_feature.py`:
   ```python
   from fastapi import APIRouter
   
   router = APIRouter(tags=["MyFeature"])
   
   @router.post("/my_endpoint")
   async def my_endpoint(data: RequestSchema) -> ResponseSchema:
       # Implementation
       pass
   ```

2. **Add schema** to `app/models/schemas.py`:
   ```python
   class RequestSchema(BaseModel):
       field: str
   
   class ResponseSchema(BaseModel):
       result: str
   ```

3. **Register router** in `app/__init__.py`:
   ```python
   from app.routes.my_feature import router as my_feature_router
   app.include_router(my_feature_router)
   ```

4. **Test** with `test_client_comprehensive.py`

### Adding a New Service

1. **Create service** in `app/services/my_service.py`:
   ```python
   class MyService:
       def __init__(self):
           pass
       
       def my_method(self):
           pass
   ```

2. **Export** in `app/services/__init__.py`:
   ```python
   from .my_service import MyService
   __all__ = ["MyService"]
   ```

3. **Use** in route handlers:
   ```python
   from app.services import MyService
   my_service = MyService()
   ```

### Adding a New Model

1. **Define model** in `app/models/neural_models.py`:
   ```python
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Architecture
       
       def forward(self, x):
           # Forward pass
           pass
   ```

2. **Create service** in `app/services/inference_service.py`:
   ```python
   class MyModelService:
       def __init__(self):
           self.model = MyModel()
           self.model.eval()
   ```

## Configuration Examples

### Development Environment
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
HOST=127.0.0.1
PORT=8000
RELOAD=true
```

### Production Environment
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
RELOAD=false
WAKEWORD_CONFIDENCE_THRESHOLD=0.8
VOICE_VERIFICATION_THRESHOLD=0.75
```

## Database Integration

To add database support:

1. Install ORM: `pip install sqlalchemy alembic`
2. Create `app/db/database.py` for connection
3. Create `app/db/models.py` for schema
4. Create `app/db/crud.py` for CRUD operations
5. Use in services via dependency injection

## Performance Considerations

1. **Audio Processing**: Most expensive operation (Mel spectrograms)
2. **Model Inference**: CPU-based, multi-thread safe
3. **File I/O**: Consider caching frequently accessed data
4. **Async**: All routes use async for concurrent handling
5. **Scaling**: Use multiple workers with Gunicorn

## Security Best Practices

✓ Input validation via Pydantic  
✓ CORS configuration  
✗ Add JWT authentication (TODO)  
✗ Add rate limiting (TODO)  
✗ Add request size limits (TODO)  
✓ Error messages don't expose internals  
✓ Logging without sensitive data  

## Testing Strategy

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test full pipelines
3. **API Tests**: Test HTTP layer
4. **Load Tests**: Test under stress

```bash
# Run comprehensive tests
python test_client_comprehensive.py

# Run pytest tests (when added)
pytest app/ -v
```

## Deployment Checklist

- [ ] All environment variables set in `.env`
- [ ] Log level set to INFO or WARNING
- [ ] CORS origins configured correctly
- [ ] Error messages don't expose sensitive data
- [ ] Rate limiting configured
- [ ] Authentication/authorization added
- [ ] HTTPS/SSL enabled
- [ ] Database backups configured
- [ ] Monitoring/alerting set up
- [ ] Health checks working
- [ ] Load balancing configured
- [ ] Container image built and tested

## Troubleshooting

### Imports not working
- Ensure Python path is correct
- Verify virtual environment is activated
- Check `__init__.py` files exist

### Models not loading
- Check PyTorch installation
- Verify CUDA drivers if using GPU
- Check model files exist in correct location

### Audio processing issues
- Verify audio is WAV format
- Check audio duration (0.5-30s)
- Ensure sample rate is 8000-48000 Hz

### Performance issues
- Monitor CPU usage
- Consider GPU acceleration
- Implement request caching
- Use async properly

## Migration Path

If migrating from old monolithic `app.py`:

1. Keep old `app.py` for reference
2. Implement same endpoints in new structure
3. Run both in parallel during testing
4. Switch routing to new implementation
5. Archive old code

## Future Enhancements

- [ ] WebSocket support for streaming audio
- [ ] Database for persistent storage
- [ ] User authentication system
- [ ] Model versioning and rollback
- [ ] A/B testing framework
- [ ] Advanced monitoring/metrics
- [ ] Real-time model retraining
- [ ] Multi-language support

---

**Architecture by**: Production-Ready Design Patterns  
**Last Updated**: 2025-11-17  
**Version**: 1.0.0
