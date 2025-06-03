# Gate Entry Automation

Handles automated warehouse gate entry using NFC technology.

## Files

- `nfc_processor.py` — NFC tag reader abstraction and gate entry business logic

## Features

- Simulated and real NFC reader support
- Automatic batch creation upon tag scan
- Integration with inventory and transaction models
- Error handling and audit logging

## Usage

The NFC processor is started as a background task in the main application.
