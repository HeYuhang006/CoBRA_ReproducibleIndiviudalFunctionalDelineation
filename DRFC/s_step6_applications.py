"""Manual entry point. See README_运行顺序.md before running."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Common'))
from f_runtime import f_cli
if __name__=='__main__':f_cli('DRFC',6)
