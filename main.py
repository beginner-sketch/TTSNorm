from __future__ import print_function
import os
import time
from datetime import datetime
import sys
import argparse
import numpy as np
import math
from models.TTSNorm import *
from utils.data_utils_mutilsource import *
from utils.math_utils_mutilsource import *
from utils.tester_mutilsource import model_inference
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

np.random.seed(1337)  # for reproducibility
torch.backends.cudnn.benchmark = True
# Params #
parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'TTSNorm', type=str, help = 'model name')
parser.add_argument('--batch_size', default = 8, type=int, help='batch size')
parser.add_argument('--test_batch_size', default = 8, type=int, help='test batch size')
parser.add_argument('--lr', default = 0.0001, type=int, help='learning rate')
parser.add_argument('--data', default = 'NYC', type=str, help = 'NYC or AIR')
parser.add_argument('--indim', default = 1, type=int, help = 'in_dim is 1 or 4')
parser.add_argument('--fusion', default = False, type=bool, help='model uses fusion or not')
parser.add_argument('--mix_loss', default = True, type=bool, help='mix loss or total loss')
parser.add_argument('-schemeA', default = 1, type=int, help = 'use schemeA or not')
parser.add_argument('-schemeB', default = 1, type=int, help = 'use schemeB or not')
parser.add_argument('-schemeC', default = 1, type=int, help = 'use schemeC or not')
parser.add_argument('-schemeD', default = 1, type=int, help = 'use schemeD or not')
parser.add_argument('-schemeE', default = 1, type=int, help = 'use schemeE or not')
parser.add_argument('-schemeF', default = 1, type=int, help = 'use schemeF or not')
parser.add_argument('-hidden_channels', default = 16, type=int, help = 'hidden channels')
parser.add_argument('-n_his', default = 16, type=int, help = 'time step in')
parser.add_argument('-n_pred', default = 3, type=int, help = 'horizon')
parser.add_argument('-scaler', default = "global_scaler", type=str, help = 'global_scaler,source_scaler or mix_scaler')
parser.add_argument('-mode', default = 'train', type=str, help='train or eval')
parser.add_argument('-version', default = 0, type=int, help='0-4')
parser.add_argument('cuda', default = 1, type=int, help='0-1')
args = parser.parse_args() 
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu")
dataset_name = args.data
layers = int(np.log2(args.n_his))
#######################################
def train(device, model, dataset, n, n_source):
    target_n = "schemeA{}_schemeB{}_schemeC{}_schemeD{}_schemeE{}_schemeF{}_hc{}_l{}_his{}_pred{}_v{}_scaler{}_fusion{}_mixloss{}".format(args.schemeA, args.schemeB, args.schemeC, args.schemeD, args.schemeE, args.schemeF, args.hidden_channels, layers, args.n_his, args.n_pred, args.version,args.scaler,args.fusion,args.mix_loss)
    target_fname = '{}_{}_{}'.format(args.model, dataset_name, target_n)
    target_model_path = os.path.join('MODEL', '{}.h5'.format(target_fname))
    print('=' * 10)
    print("training model...")
    print("releasing gpu memory....")
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    min_rmse = 1000
    min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)
    stop = 0
    nb_epoch = 500

    for epoch in range(nb_epoch):  
        starttime = datetime.now()
        model.train()
        for j, x_batch in enumerate(gen_batch(dataset.get_data('train'), args.batch_size, dynamic_batch=True, shuffle=True)):
            xh = x_batch[:, 0: args.n_his]            
            y = x_batch[:, args.n_his:args.n_his + args.n_pred,:,:,:]            
            xh = torch.tensor(xh, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            model.zero_grad()
            pred = model(xh)
            if args.mix_loss:
                var = torch.var(pred, unbiased=True)
                shrinkage = 1 - var / torch.sum(pred ** 2)
                theta = shrinkage * pred
                loss = criterion(theta, y)
            else:
                loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        if epoch % 1 == 0:
            model.eval()
            mape_val, mae_val, rmse_val, mape_test, mae_test, rmse_test = model_inference(device, model, dataset, args.test_batch_size, args.n_his, args.n_pred, min_va_val, min_val, n, args.scaler) # [horizon, sources+1]
            print(f'Epoch {epoch}:')
            for i in range(n_source):
                print('Source {}:'.format(i))
                print('|One Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                      .format(mape_val[0, i], mape_test[0, i], mae_val[0, i], mae_test[0, i], rmse_val[0, i], rmse_test[0, i]))
                print('|Two Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                      .format(mape_val[1, i], mape_test[1, i], mae_val[1, i], mae_test[1, i], rmse_val[1, i], rmse_test[1, i]))
                print('|Three Hour| MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                      .format(mape_val[2, i], mape_test[2, i], mae_val[2, i], mae_test[2, i], rmse_val[2, i], rmse_test[2, i]))
            total_rmse = rmse_val.sum()
            if total_rmse < min_rmse:
                torch.save(model.state_dict(), target_model_path)
                min_rmse = total_rmse
                stop = 0
            else:
                stop += 1
            if stop == 7:
                break
            endtime = datetime.now()
            epoch_time = (endtime - starttime).seconds
            print("Time Used:", epoch_time," seconds ")
    checkpoint = torch.load(target_model_path, map_location=torch.device('cpu'))
    model.load_my_state_dict(checkpoint)
    mape_val, mae_val, rmse_val, mape_test, mae_test, rmse_test = model_inference(device, model, dataset, args.test_batch_size, args.n_his, args.n_pred, min_va_val, min_val, n, args.scaler)
    print("Finished Training-------------------")
    print("*"*40)
    print("*"*40)
    print("*"*40)
    print(f'Epoch {epoch}:')
    for i in range(n_source):
        print('Source {}:'.format(i))
        print('|One Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mape_val[0, i], mape_test[0, i], mae_val[0, i], mae_test[0, i], rmse_val[0, i], rmse_test[0, i]))
        print('|Two Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mape_val[1, i], mape_test[1, i], mae_val[1, i], mae_test[1, i], rmse_val[1, i], rmse_test[1, i]))
        print('|Three Hour| MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mape_val[2, i], mape_test[2, i], mae_val[2, i], mae_test[2, i], rmse_val[2, i], rmse_test[2, i]))

def eval(device, n_source, model, dataset, n, versions):
    print('=' * 10)
    print("evaluating model...")
    mape_val_v, mae_val_v, rmse_val_v, mape_te_v, mae_te_v, rmse_te_v = [], [], [], [], [], []
    for _v in versions:
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)  
        target_n = "schemeA{}_schemeB{}_schemeC{}_schemeD{}_schemeE{}_schemeF{}_hc{}_l{}_his{}_pred{}_v{}_scaler{}_fusion{}_mixloss{}".format(args.schemeA, args.schemeB, args.schemeC, args.schemeD, args.schemeE, args.schemeF, args.hidden_channels, layers, args.n_his, args.n_pred, _v,args.scaler,args.fusion,args.mix_loss)
        target_fname = '{}_{}_{}'.format(args.model, dataset_name, target_n)
        target_model_path = os.path.join('MODEL', '{}.h5'.format(target_fname))        
        if os.path.isfile(target_model_path):
            print(' path is : ', target_model_path)
            model.load_my_state_dict(torch.load(target_model_path))
        else:
            print("file not exist")
            break
        print(f'Version:{_v}')
        mape_val, mae_val, rmse_val, mape_test, mae_test, rmse_test = model_inference(device, model, dataset, args.test_batch_size, args.n_his, args.n_pred, min_va_val, min_val, n, args.scaler)
        mape_val_v.append(mape_val)
        mae_val_v.append(mae_val)
        rmse_val_v.append(rmse_val)
        mape_te_v.append(mape_test)
        mae_te_v.append(mae_test)
        rmse_te_v.append(rmse_test)
        for i in range(n_source):
            print('Source {}:'.format(i))
            print('|One Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                  .format(mape_val[0, i], mape_test[0, i], mae_val[0, i], mae_test[0, i], rmse_val[0, i], rmse_test[0, i]))
            print('|Two Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                  .format(mape_val[1, i], mape_test[1, i], mae_val[1, i], mae_test[1, i], rmse_val[1, i], rmse_test[1, i]))
            print('|Three Hour| MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
                  .format(mape_val[2, i], mape_test[1, i], mae_val[2, i], mae_test[2, i], rmse_val[2, i], rmse_test[2, i]))
    mape_val = np.array(mape_val_v).mean(axis=0)   
    mae_val = np.array(mae_val_v).mean(axis=0)
    rmse_val = np.array(rmse_val_v).mean(axis=0)
    mape_test = np.array(mape_te_v).mean(axis=0)
    mae_test = np.array(mae_te_v).mean(axis=0)
    rmse_test = np.array(rmse_te_v).mean(axis=0)
    print("*"*40)
    print("*"*40)
    print("*"*40)    
    print("All Version Average: ")
    for i in range(n_source):
        print('Source {}:'.format(i))
        print('|One Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mape_val[0, i], mape_test[0, i], mae_val[0, i], mae_test[0, i], rmse_val[0, i], rmse_test[0, i]))
        print('|Two Hour  | MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mape_val[1, i], mape_test[1, i], mae_val[1, i], mae_test[1, i], rmse_val[1, i], rmse_test[1, i]))
        print('|Three Hour| MAPE: {:6.2%}, {:6.2%}; MAE: {:.2f}, {:.2f}; RMSE: {:.2f}, {:.2f};'
              .format(mape_val[2, i], mape_test[2, i], mae_val[2, i], mae_test[2, i], rmse_val[2, i], rmse_test[2, i]))

def main():
    # set cpu number
    cpu_number = 1
    os.environ ['OMP_NUM_THREADS'] = str(cpu_number)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_number)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_number)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_number)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_number)
    torch.set_num_threads(cpu_number)
    ########################################################### 
    print('=' * 10)
    print('| Model: {0} | Dataset: {1} | History: {2} | Horizon: {3} | SchemeA: {4} | SchemeB: {5} | SchemeC: {6} | SchemeD: {7} | SchemeE: {8} | SchemeF: {9}'
          .format(args.model, dataset_name, args.n_his, args.n_pred, args.schemeA, args.schemeB, args.schemeC, args.schemeD, args.schemeE, args.schemeF))
    print("version: ", args.version)
    print("channel in: ", args.indim)
    print("hidden channels: ", args.hidden_channels)
    print("layers: ", layers)
    print("fusion:", args.fusion)
    print("scaler form: ", args.scaler)
    print("mix loss: ", args.mix_loss)
    start=time.time()
    # load data
    print('=' * 10)
    print("loading data...")
    if dataset_name == 'NYC':
        n_train, n_val, n_test = 81, 5, 5
        n = 98
        n_slots = 48
        n_source = 4
        dataset = data_gen('data/NYC.h5', (n_train, n_val, n_test), n, args.n_his + args.n_pred, n_source, n_slots, args.scaler)

    print('=' * 10)
    print("compiling model...")

    model = TTSNorm(device, num_nodes=n, num_source=n_source, n_his=args.n_his, n_pred=args.n_pred, schemeA_bool=args.schemeA, schemeB_bool=args.schemeB,
                    schemeC_bool=args.schemeC, schemeD_bool=args.schemeD,schemeE_bool=args.schemeE, schemeF_bool=args.schemeF, in_dim=1, out_dim=1, 
                    channels=args.hidden_channels, kernel_size=2, layers=layers, fusion_bool=args.fusion).to(device)

    print('=' * 10)
    print("init model...")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    if args.mode == 'train':
        train(device, model, dataset, n, n_source)
    if args.mode == 'eval':
        model.eval()
        eval(device, n_source, model, dataset, n, np.arange(0,5))
    end=time.time()
    print('Running time: %s hours.'%((end-start) // 3600))
    
if __name__ == '__main__':
    main()
