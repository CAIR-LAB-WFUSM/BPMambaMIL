import os
import numpy as np
import torch
import argparse
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
from dataset.dataset_generic import Generic_MIL_Dataset
import pandas as pd

from models.MambaMIL import MambaMIL



def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_data_integrity(dataset, split_dir, k):
    for i in range(k):
        csv_path = f'{split_dir}/splits_{i}.csv'
        df = pd.read_csv(csv_path)
        test_ids = df['test'].dropna().tolist()
        
        missing_ids = []
        for slide_id in test_ids:
            if slide_id not in dataset.slide_data['slide_id'].values:
                missing_ids.append(slide_id)
        
        print(f"Fold {i} data integrity check:")
        print(f"  Total test IDs in CSV: {len(test_ids)}")
        print(f"  Missing IDs in dataset: {len(missing_ids)}")
        if missing_ids:
            print(f"  First few missing IDs: {missing_ids[:5]}")





def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    dataset = load_dataset(args)

    print("------------------------check integrity---------------------")
    check_data_integrity(dataset, args.split_dir, args.k)
    print("------------------------check integrity---------------------")

    test_results = []
    val_results = []
    all_test_metrics = {
        'auc': [], 'acc': [], 'sensitivity': [], 'specificity': [],
        'ppv': [], 'npv': [], 'f1': []
    }
    all_val_metrics = {
        'auc': [], 'acc': [], 'sensitivity': [], 'specificity': [],
        'ppv': [], 'npv': [], 'f1': []
    }
    results_dir = os.path.join(args.results_dir, args.model_type)
    os.makedirs(results_dir, exist_ok=True)

    for i in range(args.k):
        seed_torch(args.seed)

        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        print(f"\nFold {i}")
        print(f"Test Dataset samples: {len(test_dataset)}")
        print(f"Validation Dataset samples: {len(val_dataset)}")
            
        model = load_model(args, i)
        model.to(device)
        model.eval()

        # Evaluate on test set
        test_loader = get_split_loader(test_dataset, testing=True)
        test_results_fold = summary(model, test_loader, args.n_classes, i, args, 'test')
        patient_results, test_error, test_auc, acc_logger, sensitivity, specificity, ppv, npv, f1 = test_results_fold
        
        test_results.append({
            'fold': i,
            'test_auc': test_auc,
            'test_acc': 1 - test_error,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1': f1
        })
        
        # Update metrics collection
        for key, value in zip(
            ['auc', 'acc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1'],
            [test_auc, 1 - test_error, sensitivity, specificity, ppv, npv, f1]
        ):
            all_test_metrics[key].append(value)


        # Evaluate on validation set
        val_loader = get_split_loader(val_dataset, testing=True)
        val_results_fold = summary(model, val_loader, args.n_classes, i, args, 'val')
        _, val_error, val_auc, _, val_sensitivity, val_specificity, val_ppv, val_npv, val_f1 = val_results_fold
        
        val_results.append({
            'fold': i,
            'val_auc': val_auc,
            'val_acc': 1 - val_error,
            'sensitivity': val_sensitivity,
            'specificity': val_specificity,
            'ppv': val_ppv,
            'npv': val_npv,
            'f1': val_f1
        })
        
        # Update validation metrics
        for key, value in zip(
            ['auc', 'acc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1'],
            [val_auc, 1 - val_error, val_sensitivity, val_specificity, val_ppv, val_npv, val_f1]
        ):
            all_val_metrics[key].append(value)
            
        print(f'\nFold {i} Results:')
        print('Test  - Error: {:.4f}, AUC: {:.4f}'.format(test_error, test_auc))
        print('       Sensitivity: {:.4f}, Specificity: {:.4f}'.format(sensitivity, specificity))
        print('       PPV: {:.4f}, NPV: {:.4f}, F1: {:.4f}'.format(ppv, npv, f1))
        print('Val   - Error: {:.4f}, AUC: {:.4f}'.format(val_error, val_auc))
        print('       Sensitivity: {:.4f}, Specificity: {:.4f}'.format(val_sensitivity, val_specificity))
        print('       PPV: {:.4f}, NPV: {:.4f}, F1: {:.4f}'.format(val_ppv, val_npv, val_f1))

    # Save summary results with new metrics
    for split_name, results, metrics in [('test', test_results, all_test_metrics),
                                       ('val', val_results, all_val_metrics)]:
        df = pd.DataFrame(results)
        mean_values = df.mean()
        std_values = df.std()
        
        df = pd.concat([df, pd.DataFrame({
            'fold': ['mean', 'std'],
            f'{split_name}_auc': [mean_values[f'{split_name}_auc'], std_values[f'{split_name}_auc']],
            f'{split_name}_acc': [mean_values[f'{split_name}_acc'], std_values[f'{split_name}_acc']],
            'sensitivity': [mean_values['sensitivity'], std_values['sensitivity']],
            'specificity': [mean_values['specificity'], std_values['specificity']],
            'ppv': [mean_values['ppv'], std_values['ppv']],
            'npv': [mean_values['npv'], std_values['npv']],
            'f1': [mean_values['f1'], std_values['f1']]
        })], ignore_index=True)

        csv_path = os.path.join(args.results_dir,  args.model_type,f'{split_name}_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n{split_name.capitalize()} summary saved to {csv_path}")
        
        print(f'\nAverage {split_name.capitalize()} Results:')
        for metric, values in metrics.items():
            print(f'{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}')
            
def load_dataset(args):
    if args.task == 'LUAD_LUSC':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/LUAD_LUSC.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'LUAD':0, 'LUSC':1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'BRACS':
        args.n_classes = 7
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/BRACS.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'PB':0, 'IC':1, 'DCIS':2, 'N':3, 'ADH': 4,
                                              'FEA':5, 'UDH': 6 },
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'OSU':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/MYTGCA.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'L':0, 'H':1},
                                patient_strat=False,
                                ignore=[])
        
    elif args.task == 'CAM16':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CAM16res.csv',
                                data_dir= None,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'L':0, 'H':1},
                                patient_strat=False,
                                ignore=[])
                            
    else:
        raise NotImplementedError
    
    return dataset
def load_prototypes(prototype_path):
    prototypes = []
    for fold in range(3):  
        fold_prototypes = []
        for risk in ['l', 'h']:
            file_name = f"ctpnorm-p896-{risk}-4-f{fold}.pt"
            file_path = os.path.join(prototype_path, file_name)
            proto = torch.load(file_path)
            fold_prototypes.append(proto)
        prototypes.append(fold_prototypes)
    return prototypes



prototype_path = 'data/threefold_osu_over_ant4_res'
all_prototypes = load_prototypes(prototype_path)
def load_model(args, fold):
    if args.model_type == 'mean_mil':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'max_mil':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(args.in_dim, args.n_classes)
    elif args.model_type == 'att_mil':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 'trans_mil':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout=args.drop_out, act='relu')
    elif args.model_type == 's4model':
        from models.S4MIL import S4Model
        model = S4Model(in_dim = args.in_dim, n_classes = args.n_classes, act = 'gelu', dropout = args.drop_out)
    elif args.model_type == 'mamba_mil':
        # from models.MambaMIL import MambaMIL
        # model = MambaMIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)

        from models.MambaMIL import MambaMIL
        
        # Calculate samples_per_class if using class_balanced loss
        if args.loss_type == 'class_balanced':
            train_dataset = load_dataset(args).return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, fold))[0]
            # samples_per_class = calculate_samples_per_class(train_dataset)
        else:
            samples_per_class = None

        if args.use_ensemble:
            
      
            model = MambaMIL(
                in_dim=args.in_dim, 
                n_classes=args.n_classes, 
                dropout=args.drop_out, 
                act='gelu', 
                layer=args.mambamil_layer, 
                rate=args.mambamil_rate, 
                type=args.mambamil_type,
                loss_type=args.loss_type,
                samples_per_class=samples_per_class,
                prototypes=all_prototypes[fold]
                
                
            )

    



    elif args.model_type == 'mamba2_mil':
        from models.Mamba2MIL import Mamba2MIL
        model = Mamba2MIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', survival=False, layer = args.mambamil_layer)
    else:
        raise NotImplementedError

    

    ckpt_path = os.path.join(args.ckpt_path, args.exp_code+'_s1',"s_{}_checkpoint.pt".format(fold))
    
    
    print("ckpt_path",ckpt_path)
    ckpt = torch.load(ckpt_path)
    
    model.load_state_dict(ckpt, strict=False)
    model.relocate()
    model.eval()
    return model

def calculate_sensitivity_specificity(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    
    # Calculate additional metrics
    ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    return sensitivity, specificity, ppv, npv, f1

def calculate_multiclass_sensitivity_specificity(y_true, y_pred, n_classes):
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    f1s = []
    
    for i in range(n_classes):
        true_positive = np.sum((y_true == i) & (y_pred == i))
        true_negative = np.sum((y_true != i) & (y_pred != i))
        false_positive = np.sum((y_true != i) & (y_pred == i))
        false_negative = np.sum((y_true == i) & (y_pred != i))
        
        sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) > 0 else 0
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        ppvs.append(ppv)
        npvs.append(npv)
        f1s.append(f1)
    
    return (np.mean(sensitivities), np.mean(specificities), 
            np.mean(ppvs), np.mean(npvs), np.mean(f1s))

def summary(model, loader, n_classes, fold, args, split_type='test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    total_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    fold_results = []

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, A, _ ,features,contrast_features  = model(data)

        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            
        all_probs[batch_idx] = probs.squeeze()
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.cpu().numpy().item()
        
        if n_classes == 2:
            prob_class_1 = probs[0, 1] if probs.shape[1] > 1 else probs[0]
            fold_results.append({
                'slice_id': slide_id,
                'true_label': label.item(),
                'prob_class_0': 1 - prob_class_1,
                'prob_class_1': prob_class_1
            })
        else:
            fold_results.append({
                'slice_id': slide_id,
                'true_label': label.item(),
                **{f'prob_class_{i}': probs[0, i] for i in range(n_classes)}
            })
        
        patient_results.update({
            slide_id: {
                'slide_id': np.array(slide_id), 
                'prob': probs.squeeze(), 
                'label': label.item()
            }
        })
        
        error = calculate_error(Y_hat, label)
        total_error += error

    total_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs)
        predictions = (all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs) > 0.5
        sensitivity, specificity, ppv, npv, f1 = calculate_sensitivity_specificity(all_labels, predictions)
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        sensitivity, specificity, ppv, npv, f1 = calculate_multiclass_sensitivity_specificity(
            all_labels, all_preds, n_classes)

    # Save results with new metrics
    fold_df = pd.DataFrame(fold_results)
    csv_path = os.path.join(args.results_dir, args.model_type, f'fold_{fold}_{split_type}.csv')
    fold_df.to_csv(csv_path, index=False)
    print(f"Fold {fold} {split_type} results saved to {csv_path}")
    print(f"Metrics - AUC: {auc:.4f}, Error: {total_error:.4f}")
    print(f"         Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"         PPV: {ppv:.4f}, NPV: {npv:.4f}, F1: {f1:.4f}")

    return patient_results, total_error, auc, acc_logger, sensitivity, specificity, ppv, npv, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Testing Script')
    parser.add_argument('--task', type=str, default='OSU', help='Task to run')
    parser.add_argument('--model_type', type=str, default='mamba_mil', help='Type of model')
    parser.add_argument('--in_dim', type=int, default=768, help='Input dimension')
    parser.add_argument('--results_dir', type=str, default='./experiments/train/OSU', help='Results directory')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--split_dir', type=str, default='./splits/TGCA_fivefold', help='Split directory')
    parser.add_argument('--drop_out', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--mambamil_layer', type=int, default=2, help='Number of Mamba layers')
    parser.add_argument('--mambamil_rate', type=int, default=5, help='Mamba rate')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--mambamil_type', type=str, default='SRMamba', help='Type of Mamba')
    parser.add_argument('--patch_size', type=str, default='512', help='Patch size')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model')
    parser.add_argument('--ckpt_path', type=str, default='', help='Backbone model')
    parser.add_argument('--loss_type', type=str, choices=['weighted', 'focal', 'class_balanced', 'standard'], 
                        default='weighted', help='Type of loss function to use')
    parser.add_argument('--num_models', type=int, default=5, help='Number of models in ensemble')
    
    args = parser.parse_args()
    
    results = main(args)
    print("Testing finished!")