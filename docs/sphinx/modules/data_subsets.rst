Data Subsets
=============

This module provides classes for managing prepared datasets, feature metadata, and data preprocessing pipelines.

ModelDataSubset
---------------

Container for prepared datasets, targets, exposures, and feature metadata.

.. autoclass:: outboxml.data_subsets.ModelDataSubset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

DataPreprocessor
----------------

High-level data preprocessor that prepares and persists model subsets.

.. autoclass:: outboxml.data_subsets.DataPreprocessor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PickleModelSubset
-----------------

Utility for persisting and loading ModelDataSubset objects to/from pickle.

.. autoclass:: outboxml.data_subsets.PickleModelSubset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ParquetDataset
--------------

Helper to persist datasets to a Parquet file and read them back.

.. autoclass:: outboxml.data_subsets.ParquetDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ModelConfigPickle
-----------------

Persist and load ModelConfig objects to/from pickle files.

.. autoclass:: outboxml.data_subsets.ModelConfigPickle
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PrepareEngine
-------------

Abstract base class for dataset preparation engines (pandas/polars).

.. autoclass:: outboxml.data_subsets.PrepareEngine
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PandasInterface
---------------

Pandas-based preparation engine wrapping a PrepareDataset interface.

.. autoclass:: outboxml.data_subsets.PandasInterface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PolarsInterface
---------------

Polars-based preparation engine wrapping a PrepareDatasetPl interface.

.. autoclass:: outboxml.data_subsets.PolarsInterface
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
