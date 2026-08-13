import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.preprocessing import (
    load_data,
    detect_file_type,
    normalize_data,
    UnsupportedFileTypeError,
    EmptyFileError,
    InvalidSchemaError,
    ParserError,
)
from src.preprocessing.csv_parser import CSVParser
from src.preprocessing.excel_parser import ExcelParser
from src.preprocessing.json_parser import JSONParser
from src.preprocessing.text_parser import TextParser
from src.preprocessing.pdf_parser import PDFParser


# --- Fixtures ---

@pytest.fixture
def sample_csv_file(tmp_path):
    fpath = tmp_path / "tickets.csv"
    df = pd.DataFrame([
        {"id": 1, "created_at": "2026-08-01", "topic": "Dark Mode", "message": "Can we get dark mode?"},
        {"id": 2, "created_at": "2026-08-02", "topic": "Eye Strain", "message": "App causes severe eye strain."}
    ])
    df.to_csv(fpath, index=False)
    return fpath


@pytest.fixture
def sample_xlsx_file(tmp_path):
    fpath = tmp_path / "tickets.xlsx"
    df = pd.DataFrame([
        {"record_id": 10, "date": "2026-08-01", "subject": "CSV Export", "description": "Need CSV export button."},
        {"record_id": 11, "date": "2026-08-02", "subject": "Export Data", "description": "Please allow exporting reports to CSV."}
    ])
    df.to_excel(fpath, index=False, engine="openpyxl")
    return fpath


@pytest.fixture
def sample_json_file(tmp_path):
    fpath = tmp_path / "tickets.json"
    data = {
        "tickets": [
            {"ticket": 100, "time": "2026-08-01", "title": "Offline Mode", "content": "Does the app support offline mode?"},
            {"ticket": 101, "time": "2026-08-02", "title": "No Connection", "content": "Need to view data without internet."}
        ]
    }
    fpath.write_text(json.dumps(data), encoding="utf-8")
    return fpath


@pytest.fixture
def sample_txt_file(tmp_path):
    fpath = tmp_path / "tickets.txt"
    fpath.write_text(
        "# Customer feedback notes\n"
        "Dark Mode: Screen glare causes headaches late at night.\n"
        "CSV Export: I need to export data for weekly reports.\n",
        encoding="utf-8"
    )
    return fpath


@pytest.fixture
def sample_pdf_file(tmp_path):
    fpath = tmp_path / "tickets.pdf"
    can = canvas.Canvas(str(fpath), pagesize=letter)
    can.drawString(50, 700, "Dark mode support: The interface is too bright at night.")
    can.drawString(50, 650, "Eye strain: Severe eye strain when working late hours.")
    can.save()
    return fpath


# --- File Detector Unit Tests ---

def test_detect_supported_file_types(sample_csv_file, sample_xlsx_file, sample_json_file, sample_txt_file, sample_pdf_file):
    """Verify file detector correctly identifies all Tier 1 supported file extensions."""
    assert detect_file_type(sample_csv_file) == "csv"
    assert detect_file_type(sample_xlsx_file) == "excel"
    assert detect_file_type(sample_json_file) == "json"
    assert detect_file_type(sample_txt_file) == "txt"
    assert detect_file_type(sample_pdf_file) == "pdf"


def test_detect_unsupported_file_type(tmp_path):
    """Verify UnsupportedFileTypeError is raised for unsupported extensions."""
    invalid_file = tmp_path / "data.xml"
    invalid_file.write_text("<tickets></tickets>", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file extension"):
        detect_file_type(invalid_file)


def test_detect_non_existent_file():
    """Verify FileNotFoundError is raised for non-existent file path."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        detect_file_type("non_existent_path.csv")


def test_detect_zero_byte_empty_file(tmp_path):
    """Verify EmptyFileError is raised for zero-byte empty files."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(EmptyFileError, match="Input file is empty"):
        detect_file_type(empty_file)


# --- Parsers Unit Tests ---

def test_csv_parser_valid(sample_csv_file):
    """Verify CSVParser successfully parses valid CSV files."""
    parser = CSVParser()
    df = parser.parse(sample_csv_file)
    assert len(df) == 2
    assert "message" in df.columns


def test_excel_parser_valid(sample_xlsx_file):
    """Verify ExcelParser successfully parses valid Excel files."""
    parser = ExcelParser()
    df = parser.parse(sample_xlsx_file)
    assert len(df) == 2
    assert "description" in df.columns


def test_json_parser_valid(sample_json_file):
    """Verify JSONParser parses nested JSON ticket structures."""
    parser = JSONParser()
    df = parser.parse(sample_json_file)
    assert len(df) == 2
    assert "content" in df.columns


def test_json_parser_malformed_syntax(tmp_path):
    """Verify JSONParser raises ParserError on malformed JSON."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{invalid_json_syntax: 123", encoding="utf-8")
    parser = JSONParser()
    with pytest.raises(ParserError, match="Malformed JSON syntax"):
        parser.parse(bad_json)


def test_text_parser_valid(sample_txt_file):
    """Verify TextParser parses non-empty text lines into records."""
    parser = TextParser()
    df = parser.parse(sample_txt_file)
    assert len(df) == 2
    assert "message" in df.columns


def test_pdf_parser_valid(sample_pdf_file):
    """Verify PDFParser extracts text content from PDF pages."""
    parser = PDFParser()
    df = parser.parse(sample_pdf_file)
    assert len(df) >= 1
    assert "message" in df.columns


# --- Normalizer Unit Tests ---

def test_normalize_data_field_alias_mapping():
    """Verify normalizer maps field aliases (id/ticket, subject/topic, body/message) correctly."""
    raw_data = [
        {"id": 10, "date": "2026-08-01", "subject": "Glare Issue", "description": "Too bright screen."}
    ]
    df = normalize_data(raw_data)
    assert list(df.columns) == ["ticket_id", "created_at", "topic", "message"]
    assert df.iloc[0]["ticket_id"] == 10
    assert df.iloc[0]["created_at"] == "2026-08-01"
    assert df.iloc[0]["topic"] == "Glare Issue"
    assert df.iloc[0]["message"] == "Too bright screen."


def test_normalize_data_auto_generates_missing_ids_and_topics():
    """Verify normalizer auto-generates IDs and topics when omitted."""
    raw_data = [
        {"message": "Please add dark mode theme option for night users."}
    ]
    df = normalize_data(raw_data)
    assert df.iloc[0]["ticket_id"] == 1
    assert df.iloc[0]["topic"] == "Please add dark mode"
    assert df.iloc[0]["message"] == "Please add dark mode theme option for night users."


def test_normalize_data_missing_message_column_raises_error():
    """Verify InvalidSchemaError is raised when no text message column is present."""
    raw_data = pd.DataFrame([{"colA": 123, "colB": 456}])
    with pytest.raises(InvalidSchemaError, match="Could not find a valid text"):
        normalize_data(raw_data)


# --- Loader End-to-End Unit Tests ---

def test_load_data_all_tier1_formats(sample_csv_file, sample_xlsx_file, sample_json_file, sample_txt_file, sample_pdf_file):
    """Verify load_data produces standard normalized schema across all 5 supported formats."""
    expected_cols = ["ticket_id", "created_at", "topic", "message"]

    for fpath in [sample_csv_file, sample_xlsx_file, sample_json_file, sample_txt_file, sample_pdf_file]:
        df = load_data(fpath)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == expected_cols
