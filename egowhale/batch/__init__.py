"""Ray pools for the episode DAG. One heavy model stays on each GPU.

dag: stages, their pools and dependencies, and one episode's units. actors: the Ray actors.
resources: pool sizes and the checks that the cluster and host can hold them. scheduler: the run loop.
report: the boot table, batch.jsonl, and TensorBoard. Run it with `python -m egowhale.batch <config.yaml>`.
"""

import warnings

# moviepy 1.0.3 predates Python 3.12 and warns on its own regex literals. Ray workers import this package too.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"moviepy(\.|$)")
