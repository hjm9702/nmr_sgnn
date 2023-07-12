# nmr_sgnn
Source code for the paper: [Scalable Graph Neural Network for NMR Chemical Shift Prediction](https://doi.org/10.1039/D2CP04542G)

## Data
- **NMRShiftDB2** - https://nmrshiftdb.nmr.uni-koeln.de/
- The dataset can be downloaded from https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help

## Components
- **data/nmrshiftdb2_get_data.py** - data preprocessing functions
- **gnn_models/** - gnn model architectures
- **dataset.py** - data structure & functions
- **main.py** - script for overall running code
- **model.py** - training and inference functions for gnn model
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
## Citation
```
@Article{nmr_sgnn,
  title={Scalable graph neural network for {NMR} chemical shift prediction},
  author={Han, Jongmin and Kang, Hyungu and Kang, Seokho and Kwon, Youngchun and Lee, Dongseon and Choi, Youn-Suk},
  journal={Physical Chemistry Chemical Physics},
  volume={24},
  pages={26870--26878},
  year={2322},
  doi={10.1039/D2CP04542G}
}
```
