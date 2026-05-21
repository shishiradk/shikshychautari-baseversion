import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import extract_text_from_pdfs

files = sys.argv[1:]
for p in files:
    try:
        with open(p, "rb") as f:
            txt = extract_text_from_pdfs([f.read()])
        n = len((txt or "").strip())
        flag = "OK   " if n > 400 else ("THIN " if n > 50 else "EMPTY")
        print(f"{flag} {n:>7} chars  {os.path.basename(p)}")
    except Exception as e:
        print(f"ERR          {os.path.basename(p)}  -> {e!r}")
