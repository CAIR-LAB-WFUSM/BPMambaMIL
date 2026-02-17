import argparse
import os
from os.path import join, exists
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import logging
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time

def setup_logger(fold):
   
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{log_dir}/prototype_extraction_fold_{fold}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_data_loading(df, args):

    print("\nVerifying data loading:")
    print(f"Total samples in dataset: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    if not os.path.exists(args.featdir):
        raise ValueError(f"Feature directory not found: {args.featdir}")
    
    sample_slides = np.random.choice(df['slide_id'].values, min(5, len(df)), replace=False)
    print("\nChecking sample feature files:")
    for slide in sample_slides:
        feat_path = join(args.featdir, f"{slide}.pt")
        if exists(feat_path):
            try:
                features = torch.load(feat_path)
                print(f"File {slide}.pt - Shape: {features.shape}")
            except Exception as e:
                print(f"Error loading {slide}.pt: {str(e)}")
        else:
            print(f"File not found: {slide}.pt")

def pairwise_distances_cuda(x, y=None):

    x_norm = (x**2).sum(1).view(-1, 1)
    if y is None:
        y = x
        y_norm = x_norm
    else:
        y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
    return torch.clamp(dist, min=0.0)

class AntColonyOptimizer:
 
    def __init__(self, n_prototypes=100, n_ants=20, n_iterations=50, 
                 alpha=1.0, beta=2.0, evap_rate=0.1, q=1.0):
        self.n_prototypes = n_prototypes
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evap_rate = evap_rate
        self.q = q
        self.history = {
            'scores': [],
            'pheromone': [],
            'selected_indices': []
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = 1e-10
        
    def optimize(self, features, kmeans_centers):
        features = torch.tensor(features, device=self.device, dtype=torch.float32)
        kmeans_centers = torch.tensor(kmeans_centers, device=self.device, dtype=torch.float32)
        
        n_features = len(features)
        pheromone = torch.ones(n_features, device=self.device)
        
  
        distances = pairwise_distances_cuda(features, kmeans_centers)
        heuristic = 1.0 / (distances + self.epsilon)
        
        best_score = float('inf')
        best_prototypes = None
        
        for iteration in range(self.n_iterations):
            all_ant_paths = []
            all_scores = []
            
            for ant in range(self.n_ants):
                try:
                    selected_indices = self._select_features(
                        pheromone, heuristic, features, kmeans_centers
                    )
                    
                    if len(selected_indices) >= 2:
                        score = self._evaluate_solution(features[selected_indices])
                        if score < float('inf'):
                            all_ant_paths.append(selected_indices)
                            all_scores.append(score)
                            
                            if score < best_score:
                                best_score = score
                                best_prototypes = features[selected_indices].clone()
                                
                except Exception as e:
                    logging.warning(f"Error in ant {ant} iteration {iteration}: {str(e)}")
                    continue
            
            if all_ant_paths:
                pheromone = self._update_pheromone(pheromone, all_ant_paths, all_scores)
                self.history['scores'].append(best_score)
                self.history['pheromone'].append(pheromone.cpu().numpy())
                self.history['selected_indices'].append(all_ant_paths)
            else:
                logging.warning(f"No valid paths found in iteration {iteration}")
        
        if best_prototypes is None:
            raise ValueError("Optimization failed to find valid solution")
            
        return best_prototypes, self.history
    
    def _select_features(self, pheromone, heuristic, features, centers):
        selected = []
        remaining = list(range(len(features)))
        
        for i in range(len(centers)):
            if not remaining:
                break
            
            try:
         
                pheromone_values = pheromone[remaining]
                heuristic_values = heuristic[remaining, i]
                
         
                pheromone_values = torch.where(
                    torch.isfinite(pheromone_values),
                    pheromone_values,
                    torch.ones_like(pheromone_values)
                )
                heuristic_values = torch.where(
                    torch.isfinite(heuristic_values),
                    heuristic_values,
                    torch.ones_like(heuristic_values)
                )
                
          
                pheromone_values = torch.clamp(pheromone_values, min=self.epsilon)
                heuristic_values = torch.clamp(heuristic_values, min=self.epsilon)
                
        
                log_pheromone = torch.log(pheromone_values) * self.alpha
                log_heuristic = torch.log(heuristic_values) * self.beta
                
         
                logits = log_pheromone + log_heuristic
                max_logit = torch.max(logits)
                logits = logits - max_logit
                probs = torch.exp(logits)
                
            
                sum_probs = torch.sum(probs)
                if sum_probs > 0:
                    probs = probs / sum_probs
                else:
                 
                    probs = torch.ones_like(probs) / len(probs)
                
      
                if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                    raise ValueError("Invalid probabilities detected")
                
                if not torch.all(probs >= 0) or not torch.isclose(torch.sum(probs), torch.tensor(1.0)):
                    raise ValueError("Probabilities must be non-negative and sum to 1")
                
            
                probs = probs.cpu().numpy()
                
           
                try:
                    chosen_idx = np.random.choice(remaining, p=probs)
                except ValueError as e:
                    logging.warning(f"Random choice failed: {str(e)}")
                    chosen_idx = np.random.choice(remaining) 
                    
                selected.append(chosen_idx)
                remaining.remove(chosen_idx)
                
            except Exception as e:
                logging.warning(f"Error in feature selection iteration {i}: {str(e)}")
                if remaining:  
                    chosen_idx = np.random.choice(remaining)
                    selected.append(chosen_idx)
                    remaining.remove(chosen_idx)
                    
        return selected
    
    def _evaluate_solution(self, prototypes):
        if len(prototypes) < 2:
            return float('inf')
        try:
            distances = pairwise_distances_cuda(prototypes)
            distances.fill_diagonal_(float('inf'))
            min_distances = torch.min(distances, dim=1)[0]
            if torch.all(torch.isinf(min_distances)):
                return float('inf')
            return -torch.mean(min_distances).item()
        except Exception as e:
            logging.error(f"Error in evaluate_solution: {str(e)}")
            return float('inf')
    
    def _update_pheromone(self, pheromone, all_paths, scores):
        pheromone *= (1 - self.evap_rate)
        for path, score in zip(all_paths, scores):
            pheromone[path] += self.q / (score + 1e-10)
        return pheromone

class EEFOOptimizer:

    def __init__(self, n_prototypes=100, n_eels=20, n_iterations=50,
                 c_max=2.0, c_min=1.0, w_max=1.0, w_min=1.0, 
                 spiral_c1=1.0, spiral_c2=1.0, batch_size=1000):
        self.n_prototypes = n_prototypes
        self.n_eels = n_eels
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.c_max = c_max
        self.c_min = c_min
        self.w_max = w_max
        self.w_min = w_min
        self.spiral_c1 = spiral_c1
        self.spiral_c2 = spiral_c2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = 1e-10  
        
    def optimize(self, features, kmeans_centers):
        if not torch.is_tensor(features):
            features = torch.tensor(features, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(kmeans_centers):
            kmeans_centers = torch.tensor(kmeans_centers, device=self.device, dtype=torch.float32)

        if features.size(0) == 0 or features.size(1) == 0:
            raise ValueError("Empty features tensor")
        n_features = len(features)
        # n_batches = (n_features + self.batch_size - 1) // self.batch_size
        n_batches=800
        positions = torch.rand(self.n_eels, n_features, device=self.device)
        best_position = None
        best_score = float('inf')
        
        optimization_history = {
            'scores': [],
            'positions': []
        }
        
        try:
            for iteration in range(self.n_iterations):
                energy = self._calculate_energy(iteration)
                all_scores = []
                new_positions = []
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, n_features)
                    batch_features = features[start_idx:end_idx]
                    
                    if batch_features.size(0) == 0:
                        continue
                    
                    for i in range(self.n_eels):
                        if energy > 1:
                            new_pos = self._interaction_behavior(positions, i)
                        else:
                            behavior = torch.randint(3, (1,), device=self.device)[0]
                            if behavior == 0:
                                new_pos = self._resting_behavior(positions[i], iteration)
                            elif behavior == 1:
                                new_pos = self._hunting_behavior(positions[i], best_position, iteration)
                            else:
                                new_pos = self._migration_behavior(positions[i], best_position)
                        
                        new_pos = self._apply_mutation(new_pos)
                        
                
                        valid_indices = new_pos[start_idx:end_idx] > 0.5
                        if torch.any(valid_indices):
                            valid_features = batch_features[valid_indices]
                            if len(valid_features) >= 2:  
                                score = self._evaluate_solution(valid_features)
                                all_scores.append(score)
                                new_positions.append(new_pos)
                                
                                if score < best_score:
                                    best_score = score
                                    best_position = new_pos.clone()
                
                if new_positions: 
                    positions = torch.stack(new_positions)
                    
                    optimization_history['scores'].append(best_score)
                    if best_position is not None:
                        optimization_history['positions'].append(best_position.cpu().numpy())
                
            
                if len(optimization_history['scores']) > 10:  
                    recent_scores = optimization_history['scores'][-10:]
                    if all(abs(s - recent_scores[0]) < self.epsilon for s in recent_scores):
                        logging.info("Early stopping due to convergence")
                        break
                        
            if best_position is None:
                raise ValueError("Optimization failed to find valid solution")
                
            selected_features = features[best_position > 0.5]
            if len(selected_features) < 2:
                raise ValueError("Not enough features selected")
                
            return selected_features.cpu().numpy(), optimization_history
            
        except Exception as e:
            logging.error(f"Error in EEFO optimization: {str(e)}")
            raise
    
    def _interaction_behavior(self, positions, current_idx):
        mean_pos = torch.mean(positions, dim=0)
        random_pos = torch.rand_like(positions[0])
        
        if torch.rand(1).item() > 0.5:
            new_pos = positions[current_idx] + torch.randn(1).item() * (mean_pos - positions[current_idx])
        else:
            new_pos = positions[current_idx] + torch.randn(1).item() * (random_pos - positions[current_idx])
            
        return torch.clamp(new_pos, 0, 1)
    
    def _resting_behavior(self, position, iteration):
        t = iteration / self.n_iterations
        c_star = torch.cos(torch.tensor(np.pi/2 * np.sqrt(t))) * (
            self.c_max - (self.c_max - self.c_min) * iteration / self.n_iterations
        )
        
        resting_pos = position + c_star * torch.randn_like(position)
        return torch.clamp(resting_pos, 0, 1)
    
    def _hunting_behavior(self, position, best_position, iteration):
        if best_position is None:
            return position
        iteration = max(iteration, self.epsilon)  
        w1 = self.w_max * torch.exp(-torch.sqrt(torch.tensor(iteration))/(2 * (self.n_iterations/iteration)**2))
        w2 = self.w_min * torch.cos(torch.tensor(np.pi/2 * (1 - iteration/self.n_iterations)))
        
        mean_pos = torch.mean(position)
        hunting_pos = (w1 * position * mean_pos + 
                      w2 * torch.rand(1).item() * best_position)
        
        return torch.clamp(hunting_pos, 0, 1)
    
    def _migration_behavior(self, position, best_position):
        if best_position is None:
            return position
            
        distance = torch.abs(best_position - position)
        r1, r2 = torch.rand(2)
        
        spiral_pos = (self.spiral_c1 * distance * 
                     torch.exp(torch.tensor(self.spiral_c2 * r1.item())) * 
                     torch.cos(torch.tensor(2 * np.pi * r2.item())) + 
                     best_position)
                     
        return torch.clamp(spiral_pos, 0, 1)
    
    def _apply_mutation(self, position, mutation_rate=0.1):
        if torch.rand(1).item() < mutation_rate:
            mutation = torch.randn_like(position) * 0.1
            position = position + mutation
        elif torch.rand(1).item() < mutation_rate:
            position = position + torch.rand_like(position) - 0.5
            
        return torch.clamp(position, 0, 1)
    
    def _calculate_energy(self, iteration):
        r = max(torch.rand(1).item(), self.epsilon)  
        h = 1 - iteration / self.n_iterations
        return 4 * np.sin(h) * np.log(1/r)
        
    def _evaluate_solution(self, prototypes):
        if len(prototypes) < 2:
            return float('inf')
        distances = pairwise_distances_cuda(prototypes)
        distances.fill_diagonal_(float('inf'))
        return -torch.mean(torch.min(distances, dim=1)[0]).item()

class VisualizationTools:
   
    def __init__(self, save_dir='./visualizations'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        import os
        
        self.plt = plt
        self.sns = sns
        self.PCA = PCA
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_kmeans_clusters(self, features, kmeans, save_name=None):
    
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        pca = self.PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        centers_2d = pca.transform(kmeans.cluster_centers_)
        
        self.plt.figure(figsize=(12, 8))
        self.plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=kmeans.labels_, cmap='viridis', alpha=0.6)
        self.plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                        c='red', marker='x', s=200, linewidths=3)
        self.plt.title('K-means Clustering Results')
        self.plt.xlabel('First Principal Component')
        self.plt.ylabel('Second Principal Component')
        
        if save_name:
            save_path = os.path.join(self.save_dir, f'{save_name}.png')
            self.plt.savefig(save_path)
            self.plt.close()
        else:
            self.plt.show()
            
    def plot_optimization_history(self, history, algorithm='eefo', save_name=None):
        self.plt.figure(figsize=(10, 6))
        self.plt.plot(history['scores'], '-b', label='Best Score')
        self.plt.title(f'{algorithm.upper()} Optimization Progress')
        self.plt.xlabel('Iteration')
        self.plt.ylabel('Score')
        self.plt.legend()
        self.plt.grid(True)
        
        if save_name:
            save_path = os.path.join(self.save_dir, f'{save_name}.png')
            self.plt.savefig(save_path)
            self.plt.close()
        else:
            self.plt.show()
            
    def plot_feature_distribution(self, features, prototypes, algorithm='eefo', save_name=None):
   
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        if torch.is_tensor(prototypes):
            prototypes = prototypes.cpu().numpy()
            
        pca = self.PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        prototypes_2d = pca.transform(prototypes)
        
        self.plt.figure(figsize=(12, 8))
        self.plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c='blue', alpha=0.1, label='Original Features')
        self.plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], 
                        c='red', s=100, label='Selected Prototypes')
        self.plt.title(f'Feature Distribution and {algorithm.upper()} Selected Prototypes')
        self.plt.xlabel('First Principal Component')
        self.plt.ylabel('Second Principal Component')
        self.plt.legend()
        
        if save_name:
            save_path = os.path.join(self.save_dir, f'{save_name}.png')
            self.plt.savefig(save_path)
            self.plt.close()
        else:
            self.plt.show()

def process_features(features_list, n_prototypes, algorithm='eefo', visualize=False):

    # 1. 设备和批处理配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 800  # 可以通过参数传入
    
    logging.info(f"Processing features using {device}")
    logging.info(f"Total feature lists: {len(features_list)}")
    
    try:
  
        all_patches = []
        total_features = 0
        
        for features in features_list:
         
            total_features += len(features)
        
  
        all_patches = np.zeros((total_features, features_list[0].shape[1]), dtype=np.float32)
        current_idx = 0
        
      
        for features in features_list:
            batch_size = len(features)
            all_patches[current_idx:current_idx + batch_size] = features
            current_idx += batch_size
        
        logging.info(f"Total features merged: {current_idx}")
        
    
        if visualize:
            vis = VisualizationTools()
        
   
        logging.info("Starting K-means clustering...")
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
        kmeans_centers = kmeans.fit(all_patches).cluster_centers_
        
   
        kmeans_centers = torch.tensor(kmeans_centers, device=device, dtype=torch.float32)
        
        if visualize:
            vis.plot_kmeans_clusters(
                all_patches,
                kmeans,
                save_name='kmeans_clusters_osu_res_t4'
            )
        
 
        if algorithm == 'eefo':
            logging.info("Using EEFO optimizer...")
            optimizer = EEFOOptimizer(
                n_prototypes=n_prototypes,
                batch_size=batch_size
            )
            
          
            features_tensor = torch.tensor(
                all_patches, 
                device=device, 
                dtype=torch.float32
            )
            
        
            prototypes, history = optimizer.optimize(
                features_tensor,
                kmeans_centers
            )
            
            if visualize:
                vis.plot_optimization_history(
                    history,
                    algorithm='eefo',
                    save_name='eefo_optimization_osu_res_t4'
                )
                vis.plot_feature_distribution(
                    all_patches,
                    prototypes,
                    algorithm='eefo',
                    save_name='eefo_feature_distribution_osu_res_t4'
                )
                
        elif algorithm == 'aco':
            logging.info("Using ACO optimizer...")
            optimizer = AntColonyOptimizer(
                n_prototypes=n_prototypes
            )
            
        
            features_batches = []
            for i in range(0, len(all_patches), batch_size):
                batch = all_patches[i:i + batch_size]
                features_batches.append(
                    torch.tensor(batch, device=device, dtype=torch.float32)
                )
            
         
            all_prototypes = []
            for batch in features_batches:
                batch_prototypes, batch_history = optimizer.optimize(
                    batch,
                    kmeans_centers
                )
                all_prototypes.append(batch_prototypes)
            
        
            prototypes = torch.cat(all_prototypes, dim=0)
            
            if visualize:
                vis.plot_optimization_history(
                    batch_history,  
                    algorithm='aco',
                    save_name='aco_optimization_history'
                )
                vis.plot_feature_distribution(
                    all_patches,
                    prototypes.cpu().numpy(),
                    algorithm='aco',
                    save_name='aco_feature_distribution'
                )
        else:
            logging.info("Using K-means centers as prototypes...")
            prototypes = kmeans_centers
        
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
     
        if torch.is_tensor(prototypes):
            prototypes = prototypes.cpu().numpy()
        
        logging.info(f"Feature processing completed. Prototype shape: {prototypes.shape}")
        return prototypes
        
    except Exception as e:
        logging.error(f"Error in process_features: {str(e)}")
        raise


def process_fold(fold, args, df):
 
    setup_logger(fold)
    logging.info(f"Processing fold {fold}")
    

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info(f"Using device: {device}")
    
    results_dict = {'fold': fold, 'status': 'failed', 'paths': {}}
    
    try:
        split_file = f'splits/{args.task}/splits_{fold}.csv'
        if not exists(split_file):
            logging.error(f"Split file not found: {split_file}")
            results_dict['error'] = 'Split file not found'
            return results_dict
            
        split = pd.read_csv(split_file)
        train_slides = split['train'].dropna().values
        logging.info(f"Fold {fold} - Train slides count: {len(train_slides)}")
        
        if len(train_slides) == 0:
            msg = f"No training samples found in fold {fold}"
            logging.error(msg)
            results_dict['error'] = msg
            return results_dict
        

        for risk_label in ['H', 'L']:
            risk_samples = df[
                (df['slide_id'].isin(train_slides)) & 
                (df['label'] == risk_label)
            ].copy()
            # modify here for the specific label name
            
            logging.info(f"Found {len(risk_samples)} samples for {risk_label} risk group")
            
            if len(risk_samples) == 0:
                msg = f"No samples found for {risk_label} risk group in fold {fold}"
                logging.error(msg)
                results_dict['error'] = msg
                continue
                
            selected_samples = risk_samples if len(risk_samples) < 35 else risk_samples.sample(n=35, random_state=42)
            logging.info(f"Selected {len(selected_samples)} samples for processing")
            
            features_list = []
            valid_features_count = 0
            
  
            for slide in selected_samples['slide_id']:
                feat_path = join(args.featdir, f"{slide}.pt")
                if exists(feat_path):
                    try:
                   
                        features = torch.load(feat_path, map_location=device)
                        if isinstance(features, torch.Tensor):
                         
                            features = features.cpu().numpy()
                        
                        if features.size == 0:
                            logging.warning(f"Empty features found in {slide}.pt")
                            continue
                            
                        features_list.append(features)
                        valid_features_count += 1
                        logging.info(f"Loaded features from {slide}.pt, shape: {features.shape}")
                        
                     
                        if torch.cuda.is_available() and args.gpu:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logging.error(f"Error loading {feat_path}: {str(e)}")
                        continue
            
            logging.info(f"Valid features collected: {valid_features_count}")
            if valid_features_count == 0:
                msg = f"No valid features found for {risk_label} risk group in fold {fold}"
                logging.error(msg)
                results_dict['error'] = msg
                continue
                
       
            try:
                prototypes = process_features(
                    features_list, 
                    args.t, 
                    algorithm=args.algorithm,
                    visualize=args.visualize
                )
                
                save_dir = join(args.savedir, args.task)
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = join(
                    save_dir,
                    f'{args.encoder}-p{args.psize}-{risk_label.lower()}-{args.t}-f{fold}.pt'
                )
                
           
                if not torch.is_tensor(prototypes):
                    prototypes = torch.tensor(prototypes)
            
                prototypes = prototypes.cpu()
                torch.save(prototypes, save_path)
                
                if exists(save_path):
                    results_dict['paths'][risk_label] = save_path
                    logging.info(f"Successfully saved prototypes to {save_path}")
                else:
                    raise FileNotFoundError(f"Failed to save prototypes at {save_path}")
                    
            except Exception as e:
                msg = f"Error processing features for {risk_label} risk group: {str(e)}"
                logging.error(msg)
                results_dict['error'] = msg
                continue
        
        results_dict['status'] = 'success'
        return results_dict
        
    except Exception as e:
        logging.error(f"Error processing fold {fold}: {str(e)}")
        results_dict['error'] = str(e)
        return results_dict

def run_parallel(args):
    """并行处理所有fold - CUDA加速版本"""
    start_time = time.time()
    
  
    setup_blas()
    optimize_system_resources()
      
    if torch.cuda.is_available() and args.gpu:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("Using CPU only")
        args.gpu = False
    

    save_dir = join(args.savedir, args.task)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created at: {save_dir}")
    os.makedirs('logs', exist_ok=True)
    
 
    try:
        df = pd.read_csv(args.datadf)
        print("\nDataFrame head:")
        print(df.head())
        verify_data_loading(df, args)
    except Exception as e:
        print(f"Error during data verification: {str(e)}")
        return
    

    num_processes = min(3, mp.cpu_count())  
    print(f"Using {num_processes} processes")
    

    if hasattr(args, 'batch_size'):
        args.batch_size = min(args.batch_size, 500)
    
 
    try:
        ctx = mp.get_context('forkserver')
        with ctx.Pool(processes=num_processes) as pool:
            process_fold_partial = partial(process_fold, args=args, df=df)
            results = pool.map(process_fold_partial, range(5))
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        return
    

    success_count = 0
    saved_files = []
    
    print("\nProcessing Results:")
    for result in results:
        fold = result['fold']
        status = result['status']
        
        print(f"\nFold {fold} - Status: {status}")
        if status == 'success':
            success_count += 1
            for risk_label, save_path in result['paths'].items():
                print(f"  {risk_label} risk prototypes saved to: {save_path}")
                saved_files.append(save_path)
        else:
            print(f"  Failed - Error: {result.get('error', 'Unknown error')}")
    
   
    print("\nVerifying saved files:")
    for file_path in saved_files:
        if exists(file_path):
            try:
                data = torch.load(file_path)
                print(f"Successfully verified {file_path}, shape: {data.shape}")
            except Exception as e:
                print(f"Error verifying {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    

    if torch.cuda.is_available() and args.gpu:
        torch.cuda.empty_cache()
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    end_time = time.time()
    print(f"\nProcessing Summary:")
    print(f"Total folds processed successfully: {success_count}/5")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
def setup_blas():
   

    os.environ['OPENBLAS_NUM_THREADS'] = '4' 
    os.environ['MKL_NUM_THREADS'] = '4' 
    

    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    
   
    torch.set_num_threads(4)


def optimize_system_resources():
 
    import resource
    
  
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    
    try:

        new_soft = min(hard, max(soft, 4096)) 
        resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, hard))
        print(f"Process limit adjusted: soft={new_soft}, hard={hard}")
    except Exception as e:
        print(f"Warning: Could not adjust process limit: {e}")


if __name__ == '__main__':

    import multiprocessing as mp
    setup_blas()
    optimize_system_resources()
    
  
    mp.set_start_method('forkserver', force=True)
    parser = argparse.ArgumentParser(description='Parallel prototype extraction for breast cancer WSIs')
    parser.add_argument('--datadf', type=str, required=True,
                      help='Path to dataset CSV')
    parser.add_argument('--featdir', type=str, required=True,
                      help='Path to feature directory')
    parser.add_argument('--task', type=str, required=True,
                      help='Task name')
    parser.add_argument('--encoder', default='ctpnorm', type=str,
                      help='Encoder name')
    parser.add_argument('--savedir', default='./output/prototypes_eefo_t3', type=str,
                      help='Save directory')
    parser.add_argument('--psize', default=896, type=int,
                      help='Patch size')
    parser.add_argument('-t', default=100, type=int,
                      help='Number of kmeans centers')
    parser.add_argument('--seed', default=42, type=int,
                      help='Random seed')
    parser.add_argument('--algorithm', type=str, default='eefo',
                      choices=['eefo', 'aco'],
                      help='Feature selection algorithm to use')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of the optimization process')
    parser.add_argument('--gpu', action='store_true',
                      help='Enable GPU acceleration if available')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for GPU processing')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("Starting parallel prototype extraction")
    run_parallel(args)
    print("Prototype extraction complete")