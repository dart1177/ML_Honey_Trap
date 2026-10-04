"""
Tests for src/parser.py
"""

import json
import pytest
from src.parser import load_events, group_by_session, build_session, parse_logs

@pytest.fixture
def valid_log_file(tmp_path):
    p = tmp_path / "valid.json"
    data = [
        {"session": "s1", "eventid": "cowrie.session.connect"},
        {"session": "s1", "eventid": "cowrie.command.input", "input": "ls"}
    ]
    p.write_text("\n".join(json.dumps(d) for d in data))
    return p

@pytest.fixture
def malformed_log_file(tmp_path):
    p = tmp_path / "malformed.json"
    p.write_text('{"session": "s1"}\nnot a json\n{"session": "s2"}')
    return p

def test_load_events_valid(valid_log_file):
    events = load_events(str(valid_log_file))
    assert len(events) == 2

def test_load_events_missing_file():
    events = load_events("nonexistent.json")
    assert events == []

def test_load_events_malformed_lines(malformed_log_file):
    events = load_events(str(malformed_log_file))
    assert len(events) == 2

def test_group_by_session():
    events = [
        {"session": "s1"}, {"session": "s1"},
        {"session": "s2"}, {"session": "s2"}
    ]
    grouped = group_by_session(events)
    assert len(grouped) == 2
    assert len(grouped["s1"]) == 2

def test_build_session_with_commands():
    events = [
        {"eventid": "cowrie.command.input", "input": "ls"},
        {"eventid": "cowrie.command.input", "input": "whoami"},
        {"eventid": "cowrie.command.input", "input": "pwd"}
    ]
    session = build_session("s1", events)
    assert session["command_count"] == 3
    assert session["commands"] == ["ls", "whoami", "pwd"]

def test_build_session_empty():
    session = build_session("s1", [])
    assert session["command_count"] == 0
    assert session["event_count"] == 0

def test_parse_logs_end_to_end(valid_log_file):
    sessions = parse_logs(str(valid_log_file))
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s1"
    assert sessions[0]["command_count"] == 1
