# nmr_sgnn
Source code for the paper: Scalable Graph Neural Network for NMR Chemical Shift Prediction

## Data
- **NMRShiftDB2 - https://nmrshiftdb.nmr.uni-koeln.de/

## Components
- **data/nmrshiftdb2_get_data.py** - data preprocessing functions
- **gnn_models/** - gnn model architectures
- **dataset.py** - data structure & functions
- **main.py** - script for overall running code
- **model.py** - training functions for gnn model
- **train.py** - script for model training
- **util.py**

## Dependencies
- **Python**
- **Pytorch**
- **DGL**
- **RDKit**

## Run Code Example
```shell
$ python main.py --target 13C --message_passing_mode proposed --readout_mode proposed --graph_representation sparsified
```
