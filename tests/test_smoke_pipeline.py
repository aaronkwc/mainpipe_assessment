import subprocess
from pathlib import Path


def test_pipeline_smoke():
    # Run the smoke script; it handles Docker vs local Python
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / 'scripts' / 'smoke_test.sh'

    res = subprocess.run([str(script)], cwd=str(repo_root), capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    assert res.returncode == 0, f"Smoke script failed (exit {res.returncode})\nSTDERR:\n{res.stderr}"  

    out = repo_root / 'data' / 'processed' / 'tokenized_data.jsonl'
    assert out.exists() and out.stat().st_size > 0
