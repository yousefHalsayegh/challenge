import argparse, sys, time, subprocess
from pathlib import Path
from model import LocalLLM
from analysis import run_pipeline
from model import ensure_server, discover_model, server_command
ROOT = Path(__file__).resolve().parent

def main(argv=None):
    p = argparse.ArgumentParser(description='Local semantic Trend Hunter; local mode is the default.')
    p.add_argument('--input')
    p.add_argument('--as-of')
    p.add_argument('--out')
    p.add_argument('--serve', action='store_true', help='Start only the local model server')
    p.add_argument('--versions', help='Defaults to versions.csv beside the input')
    p.add_argument('--mode', choices=['local', 'offline'], default='local')
    p.add_argument('--model-path')
    p.add_argument('--server-path')
    p.add_argument('--model-revision', default='qwen4b-v2')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--ctx-size', type=int, default=4096)
    p.add_argument('--no-autostart', action='store_true')
    p.add_argument('--cache-dir', default=str(ROOT / 'cache' / 'v2'))
    p.add_argument('--max-messages', type=int, default=96, help='Semantic review budget; 0 means all eligible messages (potentially very slow).')
    p.add_argument('--batch-size', type=int, choices=[1, 2], default=2)
    p.add_argument('--no-coordination-review', action='store_true')
    a = p.parse_args(argv)
    if a.max_messages < 0 or not 1 <= a.port <= 65535 or a.ctx_size < 2048:
        p.error('Invalid budget, port or context size')
    if not a.serve and (not all([a.input, a.as_of, a.out])):
        p.error('--input, --as-of and --out are required unless --serve is used')
    if not a.versions and a.input:
        calendar = Path(a.input).resolve().parent / 'versions.csv'
        if calendar.is_file():
            a.versions = str(calendar)
    proc = handle = None
    start = time.perf_counter()
    try:
        if a.serve:
            return subprocess.call(server_command(ROOT, a.model_path, a.server_path, a.port, a.ctx_size))
        client = None
        if a.mode == 'local':
            if not a.no_autostart:
                proc, handle = ensure_server(ROOT, a.model_path, a.server_path, a.port, a.ctx_size)
            revision = a.model_revision
            try:
                model = discover_model(ROOT, a.model_path)
                st = model.stat()
                revision += f'|{model.name}|{st.st_size}|{st.st_mtime_ns}'
            except (OSError, ValueError):
                pass
            client = LocalLLM(f'http://127.0.0.1:{a.port}/v1', a.cache_dir, revision)
        report = run_pipeline(a.input, a.as_of, a.out, a.mode, client, a.versions, a.max_messages, a.batch_size, not a.no_coordination_review)
        print(f"Wrote {len(report['trends'])} trends in {time.perf_counter() - start:.1f}s to {a.out}")
        return 0
    except (OSError, ValueError, RuntimeError) as e:
        print('ERROR: ' + str(e), file=sys.stderr)
        return 1
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        if handle:
            handle.close()
if __name__ == '__main__':
    raise SystemExit(main())
