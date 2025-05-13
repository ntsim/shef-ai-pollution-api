# Sheffield AI Pollution API

A REST API built with FastAPI for forecasting pollution data in Sheffield.

## Prerequisites

- Python 3.13 ([mise](https://mise.jdx.dev/) or similar is recommended to manage versions)

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation

The API documentation will be available at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Project Structure
```
.
├── app/                 # Application source code
│   ├── __init__.py
│   ├── main.py         # FastAPI application entry point
│   ├── api/            # API routes
│   ├── models/         # Database models
│   └── core/           # Core application logic
├── tests/              # Test files
├── alembic/            # Database migrations
├── .env                # Environment variables
└── requirements.txt    # Python dependencies
```
