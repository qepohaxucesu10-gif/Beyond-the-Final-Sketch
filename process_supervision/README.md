# Process-supervision study

The core study compares Direct, Sparse B7/B19, Full25, Ordered, and Shuffled supervision. The shared implementation in `common/process_conditioned_model_and_losses.py` contains the process-conditioned encoder–decoder and its training losses. The condition directories contain the documented experiment entry points.

Configure local dataset locations and released split files before running. The public release will include the frozen configuration files required to reproduce the reported protocol.
