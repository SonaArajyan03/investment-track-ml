"""Execute the research notebook locally with the current Python interpreter."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-date', default='2026-11-10', type=date.fromisoformat,
                        help='Exclusive price-download end, YYYY-MM-DD (default: historical 2026-11-10)')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep runtime state within the chosen local output directory.
    os.environ['ITML_REVIEW_DATE'] = str(args.review_date)
    os.environ['ITML_OUTPUT_ROOT'] = str(output_dir)
    os.environ.setdefault('MPLCONFIGDIR', str(output_dir / '.matplotlib'))
    os.environ.setdefault('IPYTHONDIR', str(output_dir / '.ipython'))
    os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(output_dir / '.jupyter-runtime'))
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(Path(__file__).resolve().parent / 'notebooks/investment_track_ml.ipynb', as_version=4)
    # Requirements must already be installed. Never pip-install inside the runner.
    for cell in notebook.cells:
        if 'install' in cell.metadata.get('tags', []):
            cell.source = "print('Using packages installed from requirements.txt')"

    run_id = date.today().isoformat() + '-' + uuid.uuid4().hex[:12]
    executed_path = output_dir / (run_id + '_executed.ipynb')
    with tempfile.TemporaryDirectory(prefix='itml-kernel-') as temporary:
        kernel_root = Path(temporary)
        kernel_dir = kernel_root / 'kernels' / 'itml'
        kernel_dir.mkdir(parents=True)
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Investment Track ML', 'language': 'python',
        }))
        prior_path = os.environ.get('JUPYTER_PATH', '')
        os.environ['JUPYTER_PATH'] = str(kernel_root) + (os.pathsep + prior_path if prior_path else '')
        def on_cell_start(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                print(f'Executing cell {cell_index}', flush=True)
        try:
            NotebookClient(notebook, timeout=1800, kernel_name='itml',
                           resources={'metadata': {'path': str(output_dir)}},
                           on_cell_start=on_cell_start).execute()
        finally:
            nbformat.write(notebook, executed_path)
            print(f'Execution record: {executed_path}', flush=True)


if __name__ == '__main__':
    main()
