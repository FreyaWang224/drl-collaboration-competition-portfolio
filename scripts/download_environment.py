"""Download supplied Mac Tennis and official legacy API into ignored storage."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

root = Path(__file__).resolve().parents[1] / 'environments'
root.mkdir(exist_ok=True)
url = 'https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip'
archive = root / 'Tennis.app.zip'
if not archive.exists():
    urllib.request.urlretrieve(url, archive)
with zipfile.ZipFile(archive) as z:
    if any('..' in Path(n).parts or Path(n).is_absolute() for n in z.namelist()):
        raise ValueError('Unsafe archive')
    z.extractall(root)
(root / 'Tennis.app' / 'Contents' / 'MacOS' / 'Tennis').chmod(0o755)
api = 'https://api.github.com/repos/udacity/deep-reinforcement-learning'
commit = '561eec3ae8678a23a4557f1a15414a9b076fdfff'
tree = json.load(urllib.request.urlopen(api + '/git/trees/' + commit + '?recursive=1'))
paths = [x['path'] for x in tree['tree'] if x['type'] == 'blob' and x['path'].startswith(('python/unityagents/', 'python/communicator_objects/'))]
for p in paths:
    dest = root / 'legacy-python' / Path(p).relative_to('python')
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve('https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/' + commit + '/' + p, dest)
urllib.request.urlretrieve('https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/' + commit + '/LICENSE', root / 'legacy-python' / 'LICENSE')
manifest = dict(environment_url=url, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                legacy_source_commit=commit, files=len(paths))
(root / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest))
