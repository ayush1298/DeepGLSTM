import argparse
import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.gcn import GCNNet
from utils import *
from training import train, predicting
import time

# Configurations based on the table
# Model 1: 1st block (3 layers) + 2nd block (2 layers)
# Model 2: 1st block (3 layers) + 2nd block (2 layers) + 3rd block (1 layer) -> DeepGLSTM
# Model 3: DeepGLSTM-> 1st block(4) + 2nd block(3) + 3rd block(2) + 4th block(1) 

# Table 5 Experiments
EXPERIMENTS_TABLE5 = [
    {
        "name": "1st block^1 -> 1st block + 2nd block (# of layers 2)",
        "k_flags": [1, 2], # k1=1, k2=2
        "layer_config": [3, 2]
    },
    {
        "name": "1st block + 2nd block -> DeepGLSTM",
        "k_flags": [1, 2, 3], # k1=1, k2=2, k3=3
        "layer_config": [3, 2, 1]
    },
    {
        "name": "DeepGLSTM -> 1st block (# of layers 4) + ... + 4th block (# of layer 1)",
        "k_flags": [1, 2, 3, 4], 
        "layer_config": [4, 3, 2, 1]
    }
]

# Table 6 Experiments: Effectiveness of using the power graph
EXPERIMENTS_TABLE6 = [
    {
        "name": "1st block GCN (Input A)",
        "k_flags": [1], # Only k1
        "layer_config": [3, 2, 1] # Uses 1st element -> 3 layers
    },
    {
        "name": "2nd block GCN (Input A^2)",
        "k_flags": [2], # Only k2
        "layer_config": [3, 2, 1] # Uses 2nd element -> 2 layers
    },
    {
        "name": "3rd block GCN (Input A^3)",
        "k_flags": [3], # Only k3
        "layer_config": [3, 2, 1] # Uses 3rd element -> 1 layer
    },
    {
        "name": "DeepGLSTM (Input A, A^2, A^3)",
        "k_flags": [1, 2, 3], 
        "layer_config": [3, 2, 1]
    }
]

def run_experiment(args, exp_config):
    dataset = args.dataset
    # Prepare data
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    
    # We assume data exists, if not, user needs to run create_data.py (or we can't do much)
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('Please run create_data.py to prepare data in pytorch format!')
        return None

    train_data = TestbedDataset(root='data', dataset=dataset+'_train')
    test_data = TestbedDataset(root='data', dataset=dataset+'_test')
    
    if args.n_samples is not None:
        print(f"Subsetting to {args.n_samples} samples")
        train_data = train_data[:args.n_samples]
        test_data = test_data[:args.n_samples]
    
    TRAIN_BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size
    
    drop_last_train = len(train_data) > TRAIN_BATCH_SIZE
    drop_last_test = len(test_data) > TEST_BATCH_SIZE
    
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=drop_last_train)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=drop_last_test)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Setup model arguments
    k_flags_list = exp_config['k_flags']
    layer_config = exp_config['layer_config']
    
    k1 = 1 if 1 in k_flags_list else 0
    k2 = 2 if 2 in k_flags_list else 0
    k3 = 3 if 3 in k_flags_list else 0
    # k4 is implicit via len(layer_configs) >= 4
    
    print(f"\nRunning Experiment: {exp_config['name']}")
    print(f"Layer Config: {layer_config}, k1={k1}, k2={k2}, k3={k3}")

    model = GCNNet(k1=k1, k2=k2, k3=k3, embed_dim=128, num_layer=1, device=device, layer_configs=layer_config)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_results = {'mse': 1000}
    
    for epoch in range(args.epoch):
        train(model, device, train_loader, optimizer, epoch+1)
        G, P = predicting(model, device, test_loader)
        ret = mse(G, P)
        current_mse = ret
        
        if current_mse < best_results['mse']:
            best_results['mse'] = current_mse
            print(f'MSE improved at epoch {epoch+1}: {current_mse:.4f}')
        else:
            print(f'No improvement. Best MSE: {best_results["mse"]:.4f}')
            
    return best_results['mse']

def main():
    parser = argparse.ArgumentParser(description="Reproduce Tables from DeepGLSTM Paper")
    parser.add_argument("--dataset", type=str, default='davis', help="Dataset Name (davis, kiba)")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--n_samples", type=int, default=None, help="Subset samples for quick testing")
    parser.add_argument("--table", type=str, default='5', choices=['5', '6', 'both'], help="Which table to reproduce (5, 6, or both)")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda device")
    
    args = parser.parse_args()
    
    tables_to_run = []
    if args.table == '5' or args.table == 'both':
        tables_to_run.append(('Table 5: Effectiveness of different components', EXPERIMENTS_TABLE5))
    if args.table == '6' or args.table == 'both':
        tables_to_run.append(('Table 6: Effectiveness of using the power graph', EXPERIMENTS_TABLE6))
        
    print(f"{'='*20} Starting Reproduction {'='*20}")
    
    for table_name, experiments in tables_to_run:
        print(f"\nProcessing {table_name}")
        results = []
        for exp in experiments:
            mse_val = run_experiment(args, exp)
            results.append({
                "Name of the Models": exp['name'],
                "Mean Squared Error": f"{mse_val:.3f}" if mse_val is not None else "N/A"
            })
            
        # Print Table
        print("\n" + "="*80)
        print(f"{table_name}")
        print(f"{'Name of the Models':<80} | {'Mean Squared Error':<20}")
        print("-" * 103)
        for res in results:
            print(f"{res['Name of the Models']:<80} | {res['Mean Squared Error']:<20}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
