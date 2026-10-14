"""nhplay: small helper package for Neural Hydrology Explorer.

The heavy lifting (datasets, LSTM/Transformer models, training loop) is done by the
NeuralHydrology library. This package only adds the thin, readable layer needed for a
learning project:

- ``nhplay.paths``            project folder layout
- ``nhplay.data``             download a few CAMELS-US basins, load/inspect them
- ``nhplay.models``           run NeuralHydrology training/evaluation from Python
- ``nhplay.evaluation``       hydrology metrics (NSE, KGE, ...) explained line by line
- ``nhplay.visualization``    hydrograph and forcing plots
"""

__version__ = "0.1.0"
