# Enhancement Plan for glm-asr-docker

## Phase 1: Core Improvements

### 1. Dockerfile Modernization
- Switch to `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` base
- Keep non-root user (appuser) for security
- Add multi-stage build to reduce image size
- Add `HEALTHCHECK` directive
- Keep `CMD` and `EXPOSE` for self-documentation

### 2. Configuration Flexibility
- Add `PORT` env var support in server.py
- Add `MODEL_ID` env var for model selection
- Create `.env.example` with all configurable options

### 3. Developer Experience
- Add `Makefile` with: `build`, `run`, `shell`, `logs`, `clean`
- Add `docker-compose.yml` for one-command startup with GPU support
- Add `.dockerignore` to speed up builds

## Phase 2: Production Readiness

### 4. Server Improvements
- Add `/health` endpoint for container orchestration
- Implement graceful shutdown (SIGTERM handling)
- Add structured logging (JSON format option)

### 5. CI/CD Pipeline (`.github/workflows/`)
- `build.yml`: Build and push to GHCR on tags
- `test.yml`: Lint, basic import test, container smoke test

### 6. Documentation
- Update README with both Docker and local setup
- Add CHANGELOG.md
- Add API examples with curl

## File Structure After Enhancement

```
├── Dockerfile          # Multi-stage, secure, PyTorch base
├── docker-compose.yml  # Easy GPU deployment
├── Makefile            # Dev commands
├── .env.example        # Config documentation
├── .dockerignore
├── .github/workflows/
│   ├── build.yml
│   └── test.yml
├── server.py           # Enhanced with health check, graceful shutdown
├── requirements.txt
└── README.md           # Updated docs
```

## Priority

Start with Phase 1 (Dockerfile + config + Makefile) for immediate developer experience improvements, then add Phase 2 for production deployment readiness.
