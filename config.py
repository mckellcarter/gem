import os

# Use repo-relative paths by default
_repo_root = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(_repo_root, 'data')
log_root = os.path.join(_repo_root, 'log_root')
