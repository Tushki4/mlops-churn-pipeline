import sys
import os

sys.path.insert(0, '.')
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"