"""Put the dashboard screenshots inside the notebook as saved outputs of small code cells, so they show in Colab, Jupyter
and GitHub without the figures/ folder. In Colab the code is hidden (form view) and only the image is shown; the image
data sits in the cell, so it still works after "Run all".

Usage: python tools/embed_images.py notebook/AWTAD_EDA_and_Modeling.ipynb
"""
import base64, io, re, sys
from pathlib import Path
import nbformat as nbf
from PIL import Image

path = Path(sys.argv[1])
nb = nbf.read(path, as_version=4)
IMG = r'<img src="data:image/jpeg;base64,([^"]+)" alt="([^"]*)" width="100%">'


def jpeg_b64(raw):
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    img.thumbnail((1200, 10000))
    buf = io.BytesIO(); img.save(buf, 'JPEG', quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def image_cell(b64, title):
    cell = nbf.v4.new_code_cell(
        f'#@title {title} {{ display-mode: "form" }}\n'
        '# The screenshot is stored in this cell, so it shows without the figures/ folder.\n'
        'from base64 import b64decode\n'
        'from IPython.display import Image, display\n'
        f"display(Image(data=b64decode('{b64}'), width=1000))")
    cell.metadata.update({'cellView': 'form', 'tags': ['hide-input']})
    cell.outputs = [nbf.v4.new_output('display_data', data={'image/jpeg': b64, 'text/plain': f'<{title}>'},
                                      metadata={'image/jpeg': {'width': 1000}})]
    return cell


cells, n = [], 0
for c in nb.cells:
    if c.cell_type != 'markdown':
        cells.append(c); continue
    attachments = c.pop('attachments', {})

    def to_tag(m):            # markdown image (file or attachment) -> inline tag, handled below
        alt, ref = m.group(1), m.group(2)
        raw = (base64.b64decode(next(iter(attachments[ref[11:]].values()))) if ref.startswith('attachment:')
               else (path.parent / ref).read_bytes())
        return f'<img src="data:image/jpeg;base64,{jpeg_b64(raw)}" alt="{alt}" width="100%">'

    src = re.sub(r'!\[([^\]]*)\]\(((?:attachment:|figures/)[^)]+)\)', to_tag, c.source)
    parts = re.split(IMG, src)
    if len(parts) == 1:
        cells.append(c); continue
    if parts[0].strip():
        cells.append(nbf.v4.new_markdown_cell(parts[0].strip()))
    for i in range(1, len(parts), 3):
        b64, alt, text = parts[i], parts[i + 1], parts[i + 2].strip()
        cells.append(image_cell(b64, alt)); n += 1
        if text:
            cells.append(nbf.v4.new_markdown_cell(text))
nb.cells = cells
nbf.write(nb, path)
print('embedded', n, 'images; notebook size', round(path.stat().st_size / 1e6, 1), 'MB')
