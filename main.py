
import os, csv
import argparse

from data import nmrshiftdb2_get_data, preprocess_full_edge
from train import train

data_path = './data'
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--target', help ='13C or 1H', default='13C', type = str)
arg_parser.add_argument('--model', help ='pagtn or mpnn', default='pagtn', type = str)
arg_parser.add_argument('--embed_mode', help ='gconcat or naive', default='gconcat', type = str)
arg_parser.add_argument('--edge_mode', help ='bond or full', default='bond', type = str)
arg_parser.add_argument('--memo', help ='settings', default='', type = str)
arg_parser.add_argument('--fold_seed', default=0, type = int)
args = arg_parser.parse_args()

print('-- CONFIGURATIONS')
print(f'--- current mode: target: {args.target} model: {args.model}, embed: {args.embed_mode}, edge: {args.edge_mode}, fold_seed: {args.fold_seed}')

data_filename = os.path.join(data_path, f'nmrshiftdb2_graph_{args.target}_{args.edge_mode}.npz')

if not os.path.isfile(data_filename):
    print('no data found, preprocessing...')
    if args.edge_mode == 'bond':
        nmrshiftdb2_get_data.preprocess(args)
    elif args.edge_mode == 'full':
        preprocess_full_edge.preprocess(args)

model, test_mae = train(args)
# save result to csv
if not os.path.isfile(f'result/result_7.csv'):
    with open(f'result/result_7.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'embed_mode', 'edge_mode', 'target', 'fold_seed', 'test_mae', 'memo'])

with open(f'result/result_7.csv', 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow([args.model, args.embed_mode, args.edge_mode, args.target, args.fold_seed, test_mae, args.memo])

