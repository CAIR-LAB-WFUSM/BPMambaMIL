import copy
import numpy as np
import torch
from utils.utils import *
import os
from dataset.dataset_generic import Generic_MIL_Dataset, save_splits
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import wandb
from models.bpmambamil import  MambaMIL
from utils.loss import ClassBalancedLoss, FocalLoss, WeightedCrossEntropyLoss,calculate_samples_per_class,calculate_class_weights



import shap


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                device = ema_param.device
                ema_param.data.mul_(self.decay).add_(model_param.data.to(device), alpha=1 - self.decay)

    def to(self, device):
        self.model.to(device)
        return self

def find_func(model_name: str):
    model_name = model_name.lower()
    if model_name in ['mean_mil', 'max_mil', 'att_mil','trans_mil', 's4model','mamba_mil','mamba2_mil']:

        return train_loop, validate
    else:
        raise NotImplementedError
    

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, min_epochs=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            min_epochs (int): Minimum number of epochs to train for before considering early stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        score = -val_loss

        if epoch < self.min_epochs:
            if self.verbose:
                print(f'Epoch {epoch}: Not considering early stopping yet.')
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

            
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss




def train(datasets, cur, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\nTraining Fold {}!'.format(cur))

    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    samples_per_class = calculate_samples_per_class(train_split)
    print(f"Samples per class: {samples_per_class}")

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

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
        from models.MambaMIL import MambaMIL


        print("Creating single MambaMIL model")
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
            prototypes=all_prototypes[cur]
            
            
        )

    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')

    model.relocate()
    print('Done!')

    print('\nInit optimizer ...',end=' ')
    # optimizer = get_optim(model, args)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('Done! optimizer:',optimizer)
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 12, min_epochs=30, verbose = True)
    else:
        early_stopping = None
    print('Done!')

    print('\nSetup LR Scheduler...', end=' ')
  
    T_0 = 10  
    T_mult = 2 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=T_0,
        T_mult=T_mult,
        eta_min=1e-6  
    )
    print('Done!')
    best_val_loss = float('inf')
    best_val_auc = 0
    plateau_counter = 0
    plateau_patience = 5
    min_lr = 1e-6

    for epoch in range(args.max_epochs):

        
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer)
        val_loss, val_error, auc, stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, args.results_dir)
        
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print("current lr:", current_lr)
        wandb.log({"learning_rate": current_lr, "epoch": epoch})


    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            plateau_counter = 0
       
            torch.save(model.state_dict(), os.path.join(args.results_dir, f"s_{cur}_best_model.pt"))
        else:
            plateau_counter += 1

        if plateau_counter >= plateau_patience:
            if current_lr <= min_lr:
                print(f"Learning rate {current_lr:.2e} has reached minimum threshold. Stopping training.")
                break
            
        if stop:
            break
        


    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error






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
all_prototypes  = load_prototypes(prototype_path)


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, ema_decay=0.999, pseudo_threshold=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.



    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)



        
        
        
  
        logits, Y_prob, Y_hat, A, _ ,features,contrast_features  = model(data)
        acc_logger.log(Y_hat, label)
  
            

            
        
   
        loss = model.calculate_loss(
            logits, 
            label,
            features,
            contrast_features,
            
            
 
            
        )
        
        loss_value = loss.item()

        train_loss += loss_value

        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
    

        error = calculate_error(Y_hat, label)
        train_error += error
        loss.backward()



        optimizer.step()
        optimizer.zero_grad()
        
        # # 更新教师模型
        # teacher_model.update(model)
        
        

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        wandb.log({f"train_acc_class_{i}": acc})

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_error": train_error
    })

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)


            logits, Y_prob, Y_hat, _, _,features,contrast_features = model(data)



            loss = model.calculate_loss(
                logits, 
                label,
                features,
                contrast_features

            )

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
            acc_logger.log(Y_hat, label)

    val_error /= len(loader)
    val_loss /= len(loader)


    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        wandb.log({f"val_acc_class_{i}": acc, "epoch": epoch})

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    wandb.log({
        "val_loss_epoch": val_loss,
        "val_error_epoch": val_error,
        "val_auc": auc,
        "epoch": epoch
    })

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return val_loss, val_error, auc, True

    return val_loss, val_error, auc, False

def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    all_Y_hat = []
    all_label = []
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            
            logits, Y_prob, Y_hat, _, _,features,contrast_features = model(data)
            # loss = model.calculate_loss(logits, label, contrast_features,self_supervised_out=self_supervised_out)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds.extend(Y_hat.cpu().numpy())
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        all_Y_hat.append(Y_hat.cpu().numpy())
        all_label.append(label.cpu().numpy())

    test_error /= len(loader)
    all_Y_hat = np.concatenate(all_Y_hat)
    all_label = np.concatenate(all_label)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger