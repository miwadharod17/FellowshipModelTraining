"""Collect hand-drawn digits for the DL-7 evaluation, then build data/raw/handdrawn_digits.npz.

Step 1 (draw):   python -m train.handdrawn collect       then open  http://localhost:8000
Step 2 (build):  python -m train.handdrawn build

The page asks for one digit at a time (always the digit you have drawn least) and saves each drawing as a
280x280 PNG in data/raw/handdrawn/. data/raw is git-ignored, so the drawings stay on your computer.
Aim for 300 (about 30 per digit). Let several people draw: different handwriting is the point.

'build' converts every PNG to a 28x28 image the way MNIST was made (crop, scale longest side to 20 px,
centre by centre of mass) and writes x (N,28,28) float 0-1 white-on-black, y (N,) to
data/raw/handdrawn_digits.npz, which train_dl7.py picks up automatically. It also writes a preview picture
data/raw/handdrawn_preview.png: LOOK AT IT before you trust the numbers.
"""
import argparse
import base64
import glob
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

DIR = 'data/raw/handdrawn'
NPZ = 'data/raw/handdrawn_digits.npz'
PREVIEW = 'data/raw/handdrawn_preview.png'
TARGET = 300

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draw digits</title>
<style>
body{font-family:system-ui,sans-serif;max-width:420px;margin:0 auto;padding:16px;text-align:center;color:#222;background:#fafafa}
h1{font-size:15px;margin:0 0 4px;font-weight:600}
#prompt{font-size:56px;font-weight:700;line-height:1.1}
canvas{background:#000;border-radius:8px;touch-action:none;width:280px;height:280px;max-width:100%}
button{font-size:16px;padding:10px 16px;margin:4px;border-radius:8px;border:1px solid #999;background:#fff;cursor:pointer}
button.main{background:#2563eb;color:#fff;border-color:#2563eb}
#bar{height:8px;background:#ddd;border-radius:4px;overflow:hidden;margin:8px 0}
#fill{height:100%;background:#16a34a;width:0}
#counts span{display:inline-block;width:34px;font-size:12px}
input{font-size:14px;padding:6px;width:90px}
small{color:#666}
</style></head><body>
<h1>Draw this digit, big and centred (Enter = save, C = clear, Z = undo)</h1>
<div id="prompt">?</div>
<canvas id="c" width="280" height="280"></canvas><br>
<button class="main" id="save">Save</button><button id="clear">Clear</button>
<button id="undo">Undo last</button><button id="skip">Skip digit</button>
<div id="bar"><div id="fill"></div></div>
<div id="total"></div>
<div id="counts"></div>
<p><small>Writer: <input id="writer" value="w1"> (change it when someone else draws)</small></p>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');
let drawing=false,dirty=false,cur=0,counts=Array(10).fill(0);
function reset(){x.fillStyle='#000';x.fillRect(0,0,280,280);dirty=false}
function pos(e){const r=c.getBoundingClientRect();return[(e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height]}
c.addEventListener('pointerdown',e=>{drawing=true;c.setPointerCapture(e.pointerId);const[a,b]=pos(e);
 x.strokeStyle='#fff';x.fillStyle='#fff';x.lineWidth=20;x.lineCap='round';x.lineJoin='round';
 x.beginPath();x.arc(a,b,10,0,7);x.fill();x.beginPath();x.moveTo(a,b);dirty=true});
c.addEventListener('pointermove',e=>{if(!drawing)return;const[a,b]=pos(e);x.lineTo(a,b);x.stroke()});
['pointerup','pointercancel'].forEach(t=>c.addEventListener(t,()=>drawing=false));
function nextDigit(){const m=Math.min(...counts);for(let k=1;k<=10;k++){const d=(cur+k)%10;if(counts[d]===m)return d}return 0}
function show(j){counts=j.counts;const t=counts.reduce((a,b)=>a+b,0);
 document.getElementById('prompt').textContent=cur;
 document.getElementById('fill').style.width=Math.min(100,t/%TARGET%*100)+'%';
 document.getElementById('total').textContent=t+' / %TARGET% saved';
 document.getElementById('counts').innerHTML=counts.map((n,d)=>'<span>'+d+':'+n+'</span>').join('')}
async function post(u,b){const r=await fetch(u,{method:'POST',body:JSON.stringify(b||{})});return r.json()}
async function save(){if(!dirty)return;
 const j=await post('/save',{label:cur,writer:document.getElementById('writer').value,image:c.toDataURL('image/png')});
 cur=(function(){counts=j.counts;return nextDigit()})();show(j);reset()}
document.getElementById('save').onclick=save;
document.getElementById('clear').onclick=reset;
document.getElementById('undo').onclick=async()=>{const j=await post('/undo');show(j)};
document.getElementById('skip').onclick=()=>{cur=(cur+1)%10;document.getElementById('prompt').textContent=cur;reset()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 if(e.key==='Enter'){e.preventDefault();save()}else if(e.key==='c'||e.key==='C')reset();
 else if(e.key==='z'||e.key==='Z')document.getElementById('undo').click()});
fetch('/counts').then(r=>r.json()).then(j=>{counts=j.counts;cur=nextDigit();show(j)});reset();
</script></body></html>""".replace('%TARGET%', str(TARGET))


def counts():
    c = [0] * 10
    for p in glob.glob(f'{DIR}/*.png'):
        m = re.match(r'(\d)_', os.path.basename(p))
        if m:
            c[int(m.group(1))] += 1
    return c


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype='application/json'):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/counts':
            self._send(json.dumps({'counts': counts()}))
        else:
            self._send(PAGE, 'text/html; charset=utf-8')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or '{}')
        os.makedirs(DIR, exist_ok=True)
        if self.path == '/save':
            label = int(body['label'])
            writer = re.sub(r'[^A-Za-z0-9]', '', body.get('writer', 'w')) or 'w'
            png = base64.b64decode(body['image'].split(',', 1)[1])
            name = f'{DIR}/{label}_{writer}_{int(time.time() * 1000)}.png'
            with open(name, 'wb') as f:
                f.write(png)
        elif self.path == '/undo':
            files = sorted(glob.glob(f'{DIR}/*.png'), key=os.path.getmtime)
            if files:
                os.remove(files[-1])
        self._send(json.dumps({'counts': counts()}))


def collect(port):
    print(f'Open http://localhost:{port} in your browser. Press Ctrl+C here when you are done.')
    print('Saving into', os.path.abspath(DIR))
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()


# ---------------------------------------------------------------- build
def mnist_style(img):
    """PIL 'L' image, white strokes on black -> 28x28 float32 0-1, made like MNIST."""
    from PIL import Image
    from scipy import ndimage
    a = np.asarray(img, dtype=np.float32) / 255.0
    ys, xs = np.where(a > 0.1)
    if len(ys) == 0:
        return None
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = np.asarray(Image.fromarray((crop * 255).astype('uint8')).resize((nw, nh), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    out = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top:top + nh, left:left + nw] = small
    cy, cx = ndimage.center_of_mass(out)
    return np.clip(ndimage.shift(out, (14 - cy, 14 - cx), order=1, mode='constant'), 0, 1).astype(np.float32)


def build():
    from PIL import Image
    X, Y = [], []
    for p in sorted(glob.glob(f'{DIR}/*.png')):
        m = re.match(r'(\d)_', os.path.basename(p))
        if not m:
            continue
        im = mnist_style(Image.open(p).convert('L'))
        if im is None:
            print('skipping empty drawing', p)
            continue
        X.append(im)
        Y.append(int(m.group(1)))
    assert X, f'No drawings found in {DIR}. Run "collect" first.'
    X, Y = np.stack(X), np.array(Y, dtype=np.int64)
    np.savez(NPZ, x=X, y=Y)
    print(f'saved {len(Y)} digits to {NPZ}   per digit: {np.bincount(Y, minlength=10).tolist()}')
    if len(Y) < TARGET:
        print(f'NOTE: only {len(Y)} of {TARGET} so far.')

    # preview sheet: first 20 per row, enlarged 2x, sorted by label
    order = np.argsort(Y, kind='stable')
    cols = 20
    rows = int(np.ceil(len(Y) / cols))
    sheet = np.zeros((rows * 28, cols * 28), np.float32)
    for k, i in enumerate(order):
        r, c = divmod(k, cols)
        sheet[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = X[i]
    Image.fromarray((sheet * 255).astype('uint8')).resize((cols * 56, rows * 56), Image.NEAREST).save(PREVIEW)
    print('preview picture:', PREVIEW)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['collect', 'build'])
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    collect(a.port) if a.command == 'collect' else build()
