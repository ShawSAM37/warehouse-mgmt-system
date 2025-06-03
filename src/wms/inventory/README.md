# Inventory Management

Core inventory logic, models, and validation for the WMS.

## Files

- `models.py` — SQLAlchemy ORM models for all warehouse entities
- `schemas.py` — Pydantic schemas for API validation
- `fifo.py` — FIFO batch consumption logic

## Features

- Robust data models and relationships
- Strict input/output validation
- FIFO logic for batch-wise stock consumption
- Used by API, gate entry, and replenishment modules

