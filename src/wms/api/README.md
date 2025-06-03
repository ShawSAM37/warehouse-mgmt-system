# API Endpoints

This module contains all FastAPI route definitions for the Warehouse Management System.

## Structure

- `endpoints.py` — Main router, includes CRUD and business logic endpoints for products, batches, transactions, suppliers, purchase orders, and more.

## Features

- RESTful endpoints for all core entities
- Input/output validation using Pydantic schemas
- Error handling and pagination
- Integration with business logic modules (FIFO, Replenishment, Gate Entry)

## Usage

Routers from this module are included in the main FastAPI app (`main.py`).

