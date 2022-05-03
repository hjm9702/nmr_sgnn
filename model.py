import numpy as np
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import MC_dropout
from sklearn.metrics import mean_absolute_error

        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, n_forward_pass = 5, cuda = torch.device('cuda:0')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-10)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

    max_epochs = 500
    val_log = np.zeros(max_epochs)
    
    for epoch in range(max_epochs):
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs, n_nodes, shifts, masks = batchdata
            
            shifts = (shifts[masks] - train_y_mean) / train_y_std
            
            inputs = inputs.to(cuda)
            n_nodes = n_nodes.to(cuda)
            shifts = shifts.to(cuda)
            masks = masks.to(cuda)
            
            #predictions = net(inputs, n_nodes, masks)
            
            pred = net(inputs, n_nodes, masks)
            # Modified Loss
            loss = torch.abs(pred - shifts).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item() * train_y_std

        #print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  train_size, train_size, train_loss, (time.time()-start_time)/60))
    
        # validation
        val_y = np.hstack([inst[-2][inst[-1]] for inst in iter(val_loader.dataset)])
        val_y_pred = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
        val_loss = mean_absolute_error(val_y, val_y_pred)
        
        val_log[epoch] = val_loss
        if epoch % 10 == 0: print('--- validation epoch %d, processed %d, train_loss %.3f, current MAE %.3f, best MAE %.3f, time elapsed(min) %.2f' %(epoch, val_loader.dataset.__len__(), train_loss, val_loss, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
        
        lr_scheduler.step(val_loss)
        
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            break

    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    net.eval()
    MC_dropout(net)
    tsty_pred = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
            inputs = batchdata[0].to(cuda)
            n_nodes = batchdata[1].to(cuda)
            masks = batchdata[3].to(cuda)

            mean_list = []
            var_list = []

            for _ in range(n_forward_pass):
                mean = net(inputs, n_nodes, masks)
                mean_list.append(mean.cpu().numpy())
            
            tsty_pred.append(np.array(mean_list).transpose())

    tsty_pred = np.vstack(tsty_pred) * train_y_std + train_y_mean
    
    return np.mean(tsty_pred, 1)
