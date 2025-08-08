#!/usr/bin/env python3
"""
Query Matcher - Simple Entry Point
Start the Query Matcher with one command
"""

import subprocess
import sys

def main():
    print("Starting Query Matcher...")
    print("-" * 40)
    
    try:
        # Run the menu script
        subprocess.run([sys.executable, "menu.py"])
    except KeyboardInterrupt:
        print("\nQuery Matcher stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()