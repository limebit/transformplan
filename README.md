# transformplan

## Setup

### Installation

This command will create a virtualenv and install all dependencies

`make install`

### Lint

This command will ensure all dependencies are installed and run ruff

`make lint`

### Format

This command will ensure all dependencies are installed and run ruff format

`make format`

### Cleanup

This command will delete the virtualenv

`make clean`

## Development

- Run python script: `uv run python <filename.py>`, e.g. `uv run python train.py`
- Add new dependency: `uv add <package>`, e.g. `uv add scikit-learn`
- Add dev dependency: `uv add --group dev <package>`

## Docker

- build the docker container using `docker-compose build` (You need to make sure that docker has enough memory to build the image)
- start the jupyter lab docker container using `docker-compose up`
- Copy the link (incl. token) from the console and paste it into the browser

## Data Management (DVC)

This project uses [DVC](https://dvc.org) for data versioning with Hetzner Storage Box as the remote storage.

### Prerequisites

Ensure you have the Hetzner SSH key at `~/.ssh/hetzner_storagebox`
(This is set up automatically via JumpCloud on company devices)

### Daily Workflow

```bash
# Add new or modified data file
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
git commit -m "Update dataset"

# Push code and data (dvc push runs automatically via pre-push hook)
git push
```

### After Cloning

```bash
git clone <repo-url>
cd <repo>
make install-dev
dvc pull  # Required to download data files from remote storage
```

### Pre-commit Hooks

This template includes DVC pre-commit hooks (installed automatically via `make install-dev`):

- **dvc-pre-commit**: Runs `dvc status` before commits to show the state of DVC-tracked files. If it shows changed files, you should run `dvc add <file>` and amend your commit (or create a follow-up commit) to include the updated `.dvc` pointer.
- **dvc-pre-push**: Runs `dvc push` before `git push` to ensure data is uploaded to remote storage.
- **dvc-post-checkout**: Runs `dvc checkout` after `git checkout` to sync local data files with the checked-out branch.
