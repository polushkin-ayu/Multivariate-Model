"""Validate saved scientific values and archived figure assets."""
from pathlib import Path
import json,re,hashlib,math
ROOT=Path(__file__).resolve().parents[1]

def tables(text):
    out=[];rows=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):rows.append(line)
        elif rows:out.append(rows);rows=[]
    return out

def values(line,ru):
    line=line.replace('{,}',',').replace('−','-')
    if ru:
        line=re.sub(r'(?<=\d)[ \u00a0\u202f](?=\d{3}(?:\D|$))','',line)
        line=re.sub(r'(?<=\d),(?=\d)', '.',line)
    else:line=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',line)
    return re.findall(r'[-]?\d+(?:\.\d+)?',line)

def check():
    v=json.loads((ROOT/'generated/numbers.json').read_text())
    sig=.55**2
    assert math.isclose(v['baseline_s'],math.sqrt((sig+.5)/sig**2),abs_tol=1e-12)
    assert math.isclose(v['raw_sd_N64'],.6387045276100712,abs_tol=1e-12)
    assert v['crossing']==[311,362]
    assert round(v['rows'][0]['full_gain_per_100000'])==-8711
    manifest=json.loads((ROOT/'provenance/manifest.json').read_text())
    assert len(manifest['outputs'])==32
    for row in manifest['outputs']:
        p=ROOT/row['path'];assert p.exists(),p
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
        for name in row['source_inputs']+row['generators']:assert (ROOT/name).exists(),name
    summary=dict(logical_figures=8,figure_assets=32,analytic_checks='passed',asset_hashes='passed')
    assert not (ROOT/'manuscript').exists() and not (ROOT/'paper-en').exists()
    (ROOT/'provenance/checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':check()
