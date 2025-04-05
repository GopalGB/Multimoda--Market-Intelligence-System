# tests/test_multimodal.py
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_fusion import MultimodalMarketIntelligence

def main():
    print("Initializing multimodal system...")
    # Initialize the multimodal system
    mmi = MultimodalMarketIntelligence()

    # Rest of the code remains the same
    # ...