from fastmcp import FastMCP
import yaml
import os
import requests
import argparse
import json
from pathlib import Path
from typing import List
from PIL import Image
from io import BytesIO
import sys

with open(f"./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if config.get("proxy_on", False):
    os.environ["http_proxy"] = config.get("HTTP_PROXY", "http://127.0.0.1:7890")
    os.environ["https_proxy"] = config.get("HTTPS_PROXY", "http://127.0.0.1:7890")

# MCP client
class MindBrush:
    def __init__(self, config_path: str = "./config.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.servers_info = self.config.get("servers", {})
        self.temp_dir_path = Path(self.config.get("temp_dir", "./temp")).absolute()
        self.temp_dir_path.mkdir(parents=True, exist_ok=True)
        