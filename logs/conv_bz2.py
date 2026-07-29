import bz2
import re
import shutil
from pathlib import Path

src = Path("dataset.xlm.bz2")
out = Path("dataset.txt")

# Mode le plus rapide: False = juste decompresser
strip_xml_tags = True

if not strip_xml_tags:
    with bz2.open(src, "rb") as f_in, out.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=4 * 1024 * 1024)
else:
    tag_re = re.compile(r"<[^>]+>")
    with bz2.open(src, "rt", encoding="utf-8", errors="replace") as f_in, \
         out.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in, 1):
            f_out.write(tag_re.sub("", line))
            if i % 200000 == 0:
                print(f"{i} lignes traitees...")

print(f"Termine: {out}")
