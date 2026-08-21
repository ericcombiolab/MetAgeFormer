import os
import anndata as ad
import numpy as np
import torch
from metageformer_torch.models import MetAgeFormer_Pretrained
from metageformer_torch.dataset import load_dataset_from_adata_NMR
import pandas as pd
from tqdm import tqdm
from permetrics.regression import RegressionMetric
import json
import argparse
import pickle

from utils import *


def test_imputation(
    model,
    dataloader,
    test_adata,
    n_keep:int=50,
    device:str='cpu'
    ):

    # ignore pandas copywarning
    pd.options.mode.chained_assignment = None   

    # switch model into evalution mode
    model.eval() 

    # anndata collection of each mini-batch
    test_adata_collect = [] 


    # z-score params
    z_score_mean = test_adata.var['Z-score mean'].values.ravel() 
    z_score_std = test_adata.var['Z-score std'].values.ravel() 


    count = 0


    for data in tqdm(dataloader, desc='testing'):

        # if no pre-defined 'test_mask_nkeep'
        if 'test_mask' not in data.layers:
            X, idx_masked = keep_nonNaN_values(data.layers['Z-score normalized'].copy(), n=n_keep, random_seed=42) 
            X = pd.DataFrame(X)
            data.layers['test_mask'] = idx_masked             # mask matrix
        else:
            idx_masked = data.layers['test_mask']              
            X = data.layers['Z-score normalized'].copy()
            X[idx_masked] = np.nan
    

        inputs, _ = Tokenizer.tokenize_from_anndata(data, padding='longest', masking='specify', data_layer='Z-score normalized', 
                                                        masking_specify=idx_masked, mode='inference',
                                                        return_tensor=True, device=device)
         
 
        # model forward
        with torch.no_grad():
            outputs = model(inputs)
        
 
            
        logit = outputs['logit_conc'][:, 1:]                                  # drop the result for <CLS> token 

        # predicted result 
        pred = logit.cpu().detach().numpy()    
        pred[~idx_masked] = X.values[~idx_masked]                             # fill exisiting concentration values

        pred = pred * z_score_std.astype('float') + z_score_mean.astype('float')

        
        data.layers['test_imputed'] = pred
        test_adata_collect.append(data)


    # all test results
    test_adata_updated = ad.concat(test_adata_collect)
    test_adata_updated.var = test_adata.var

 
    y_hat = test_adata_updated.layers['test_imputed'].copy() 
    y = test_adata_updated.X.copy() 
    y_mask = test_adata_updated.layers['test_mask'].copy() 
   
    
    # NRMSE & RMAE for each metabolite
    nrmse_collect = []
    rmae_collect = []
    for i in range(y_hat.shape[1]):

        y_pred = y_hat[:,i][y_mask[:, i]]
        y_label = y[:,i][y_mask[:, i]]
        
        if len(y_label) < 2:                      # avoid error in calculating NRMSE
            nrmse_collect.append(0)
            rmae_collect.append(0)

        else:
            evaluator = RegressionMetric(y_label, y_pred)
            nrmse = evaluator.normalized_root_mean_square_error()
            nrmse_collect.append(nrmse)

            rmae_collect.append( np.mean( np.abs(y_label- y_pred) / y_label ) )
            
              
    metabo_imputed_metrics = pd.DataFrame({'NRMSE':nrmse_collect, 'RMAE':rmae_collect})
    metabo_imputed_metrics.index = test_adata.var_names
    

    return metabo_imputed_metrics

 
 
 
def extract_embs(
    model,
    test_dataloader,
    data_layer:str ='Z-score normalized',
    device:str='cpu'
    ):
    '''
    sample embedding extraction
    '''
    # switch model into evalution mode
    model.eval() 
    
    embs_collect = []
  
    
    count = 0
    for data in tqdm(test_dataloader, desc='testing: embedding extraction'):


        # metabolite tokenization       
        inputs, _ = Tokenizer.tokenize_from_anndata(data, padding='longest', masking='missing', 
                                                    data_layer=data_layer, mode='inference',
                                                    return_tensor=True, device=device)
        
        # model forward
        with torch.no_grad():
            outputs = model(inputs)
    

        # sample embeddings 
        embs = outputs['embs']
        embs_collect.append( embs.cpu().detach().numpy() )    


    # embeddings 
    emb_dict = {}
    embs_all = np.concatenate(embs_collect, axis=0)
    emb_dict['sample'] = embs_all


    return emb_dict





def test_attn(
    model,
    test_dataloader,
    save_dir:str='./',
    device:str='cpu'
    ):

    # ignore pandas copywarning
    pd.options.mode.chained_assignment = None   

    # switch model into evalution mode
    model.eval() 
    

    # Initialize variables for online averaging
    attn_scores_sum = []
    num_samples = 0

    for data in tqdm(test_dataloader, desc='testing: collecting attention scores'):
    
        # metabolite tokenization 
        inputs, _ = Tokenizer.tokenize_from_anndata(data, padding='longest', masking='missing', 
                                                    data_layer='Z-score normalized', mode='inference',
                                                    return_tensor=True, device=device)
         
        # model forward
        with torch.no_grad():
            outputs = model(inputs)
            
        # Get batch size
        batch_size = data.n_obs
        
        # Accumulate attention scores for each layer (online averaging)
        if not attn_scores_sum:
            # Initialize sum with first batch's mean
            attn_scores_sum = [np.mean(attn.cpu().numpy(), axis=0) * batch_size for attn in outputs['attn']]
        else:
            # Add current batch's sum to accumulated sum
            for i, attn in enumerate(outputs['attn']):
                batch_mean = np.mean(attn.cpu().numpy(), axis=0)
                attn_scores_sum[i] += batch_mean * batch_size
        
        num_samples += batch_size

    # Calculate final average
    attn_scores_avg = [attn_sum / num_samples for attn_sum in attn_scores_sum]

    # Save each tensor in the list as a separate .npy file   
    for i, attn in enumerate(attn_scores_avg):
        npy_file_path = os.path.join(save_dir, f'attn_layer_{i}.npy')
        np.save(npy_file_path, attn)

if __name__ == '__main__':

    

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--model_dir', type=str, help='.')
    parser.add_argument('--save_dir', type=str, help='.')
    parser.add_argument('--data_path', type=str, default=None, help='.')
    parser.add_argument('--batch_size', type=int, default=128,required=False, help='.')

    args = parser.parse_args()

    # loading setting file
        
    model_dir = args.model_dir
    model_config = os.path.join(model_dir, 'config.json')
    if os.path.exists(model_config):
        with open(model_config, 'r') as file:
            model_config = json.load(file)
            
    # params for model inference
    batch_size = args.batch_size                
    data_path = args.data_path
    save_dir = args.save_dir

    create_directory(save_dir)                              # the dir of evaluation result

    set_seeds(3047)                                         # random seed

    device = "cuda" if torch.cuda.is_available() else "cpu" 


    # load dataset
    if isinstance(data_path,str):
        train_loader, train_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'train.h5ad'),
                                                            shuffle=False,
                                                            batch_size=batch_size,
                                                            device=device)
        val_loader, val_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'val.h5ad'),
                                                            shuffle=False,
                                                            batch_size=batch_size,
                                                            device=device)
        test_loader, test_adata = load_dataset_from_adata_NMR(os.path.join(data_path, 'test.h5ad'),
                                                            shuffle=False,
                                                            batch_size=batch_size,
                                                            device=device)


        
    # load tokenizer
    tokenizer_path = os.path.join(model_dir,'tokenizer.pkl')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"The file of pre-trained tokenizer dose not exist in '{tokenizer_path}'. ")
    Tokenizer = load_tokenizer(tokenizer_path)
    
    # load model
    EmbeddingModule_conf = {
                "n_vocabs": {'identifier': Tokenizer.vocab_size_identifiers}
                }

    model_config['average_attn_weights'] = False   # obtain attention scores from each head (without averaging)

    model_weights_path = os.path.join(model_dir, 'model_weights.pth')
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"The file of model weight dose not exist in '{model_weights_path}'. ")
    
    
    test_model = MetAgeFormer_Pretrained(EmbeddingModule_conf, model_config, model_weights_path)
    test_model.to(device)  
     
       
    ######### 1. Pre-trained #########
    print('\n')
    print("Starting evaluation...imputation performance") 
    
    for remain_ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
        
        print(f" Imputation with {int(remain_ratio*100)}% non-missing values")  
        
        n_keep = int(test_adata.n_vars * remain_ratio)          # X% non-missing values kept during imputation evaluation
        test_imp_metrics= test_imputation(test_model, 
                                        test_loader, 
                                        test_adata,
                                        n_keep=n_keep, 
                                        device=device)
        
        impute_save_dir = os.path.join(save_dir, 'imputation', f'Ratio_{int(remain_ratio*100)}')
        create_directory(impute_save_dir)
        test_imp_metrics.to_csv( os.path.join(impute_save_dir, 'nrmse_rmae_test.csv') )
    

    
    print('\n')
    print("Starting evaluation...embedding extraction") 

    train_embs = extract_embs(test_model, train_loader, device=device)
    val_embs = extract_embs(test_model, val_loader, device=device)     
    test_embs = extract_embs(test_model, test_loader, device=device) 
          
    extract_emb_save_dir = os.path.join(save_dir, 'embeddings')
    create_directory(extract_emb_save_dir)
    with open(os.path.join(extract_emb_save_dir, 'embeddings_train.pkl'), 'wb') as file:
        pickle.dump(train_embs, file)   
    with open(os.path.join(extract_emb_save_dir, 'embeddings_test.pkl'), 'wb') as file:
        pickle.dump(test_embs, file)     
    with open(os.path.join(extract_emb_save_dir, 'embeddings_val.pkl'), 'wb') as file:
        pickle.dump(val_embs, file)         
      
      
    print('\n')
    print("Starting evaluation...attention scores") 
    
    sub_save_dir = os.path.join(save_dir, 'attn_scores')
    create_directory(sub_save_dir) 
    test_attn(
            test_model,
            test_loader, 
            save_dir=sub_save_dir,
            device=device
            )
               

