import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, TensorDataset 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load complete MNIST dataset
full_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

# Load the official test set
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True,
    transform=transform
)

# Split training data into train and validation sets
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders with customizable batch size
def create_data_loaders(batch_size=64, num_workers=0):
    """Create DataLoaders for train, validation, and test sets"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return statistics"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    """Evaluate model on given data loader"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def plot_metrics(metrics):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(metrics['train_loss'], label='Train Loss')
    ax1.plot(metrics['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(metrics['train_acc'], label='Train Acc')
    ax2.plot(metrics['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device='cuda', verbose=True):
    """
    Train the model for multiple epochs with validation
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer for training
    num_epochs : int, default=10
        Number of epochs to train
    device : str, default='cuda'
        Device to train on ('cuda' or 'cpu')
    verbose : bool, default=True
        Whether to print progress
        
    Returns:
    --------
    dict
        Dictionary containing training history
    """
    # Initialize metrics dictionary
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
    
    return metrics

def test_random_image(model, test_loader, device):
    """
    Test model on a random image from the test set
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model to test
    test_loader : DataLoader
        DataLoader containing test data
    device : torch.device
        Device to run inference on
    """
    # Get a random batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Select random image from batch
    idx = torch.randint(0, images.shape[0], (1,)).item()
    image = images[idx]
    label = labels[idx]
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot image
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.set_title(f'True Label: {label}\nPredicted: {pred}')
    ax1.axis('off')
    
    # Plot probabilities as bar chart
    classes = range(10)
    ax2.bar(classes, probabilities.cpu().numpy())
    ax2.set_xticks(classes)
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Probability')
    ax2.set_title('Model Predictions')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_result(image, output=None, prediction=None):
    """
    Visualize MNIST image with either raw output values or softmax predictions
    
    Parameters:
    -----------
    image : torch.Tensor or numpy.ndarray
        Input image of shape [1, 28, 28] or [28, 28]
    output : numpy.ndarray, optional
        Raw output values of shape [10]
    prediction : numpy.ndarray, optional
        Prediction values to apply softmax to, shape [10]
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot image
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.axis('off')
    
    # Determine what to plot on right side
    if prediction is not None:
        # Use softmax for predictions
        values = torch.nn.functional.softmax(torch.tensor(prediction), dim=0).numpy()
        ylabel = 'Probability'
        title = f'Predicted Class: {np.argmax(values)}'
    elif output is not None:
        # Use raw outputs
        values = output
        ylabel = 'Output Value'
        title = 'Raw Network Outputs'
    else:
        raise ValueError("Either output or prediction must be provided")
    
    # Plot values as bar chart
    classes = range(10)
    bars = ax2.bar(classes, values)
    ax2.set_xticks(classes)
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel(ylabel)
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def collect_activations(model, data_loader, device='cuda'):
    """
    Collect hidden layer activations from the network during forward pass
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    data_loader : DataLoader
        DataLoader containing the input data
    device : str, default='cuda'
        Device to run inference on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (inputs, activations, outputs) where activations only include hidden layers
    """
    all_inputs = []
    all_activations = []
    all_outputs = []
    activation_dict = {}
    
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for hidden layers only (exclude first and last layer)
    hooks = []
    layers = list(model.model)
    for layer in layers[1:-1]:  # Skip first and last layer
        hooks.append(layer.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            all_inputs.append(inputs.cpu().numpy())
            
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            
            # Process hidden layer activations
            batch_activations = []
            for i in range(len(activation_dict)):
                batch_activations.append(activation_dict[i])
            
            if batch_activations:  # Only if we have hidden activations
                all_activations.append(np.concatenate(
                    [act.reshape(act.shape[0], -1) for act in batch_activations], 
                    axis=1
                ))
            
            activation_dict.clear()
    
    for hook in hooks:
        hook.remove()
    
    inputs = np.concatenate(all_inputs, axis=0)
    activations = np.concatenate(all_activations, axis=0) if all_activations else np.array([])
    outputs = np.concatenate(all_outputs, axis=0)
    
    return inputs, activations, outputs

def collect_activations_embed(model, data_loader, device='cuda'):
    """
    Collect hidden layer activations from the network during forward pass
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    data_loader : DataLoader
        DataLoader containing the input data
    device : str, default='cuda'
        Device to run inference on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (inputs, activations, outputs) where activations only include hidden layers
    """
    all_inputs = []
    all_activations = []
    all_outputs = []
    activation_dict = {}
    
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for all hidden layers
    hooks = []
    
    # Add hooks for feature extraction layers (except first)
    feature_layers = list(model.features)[1:]  # Skip first layer
    for layer in feature_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Add hooks for embedding layers
    for layer in model.embedding:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Add hooks for classifier layers (except last)
    classifier_layers = list(model.classifier)[:-1]  # Skip last layer
    for layer in classifier_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            all_inputs.append(inputs.cpu().numpy())
            
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            
            # Process hidden layer activations
            batch_activations = []
            for i in range(len(activation_dict)):
                act = activation_dict[i]
                # Flatten if needed
                if len(act.shape) > 2:
                    act = act.reshape(act.shape[0], -1)
                batch_activations.append(act)
            
            if batch_activations:  # Only if we have hidden activations
                all_activations.append(np.concatenate(batch_activations, axis=1))
            
            activation_dict.clear()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate results
    inputs = np.concatenate(all_inputs, axis=0)
    activations = np.concatenate(all_activations, axis=0) if all_activations else np.array([])
    outputs = np.concatenate(all_outputs, axis=0)
    
    return inputs, activations, outputs

def collect_single_example_activations_embed(model, loader, index=None, device='cuda'):
    """
    Collect hidden layer activations for a single example from EmbedMNISTNet
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    loader : DataLoader
        DataLoader containing the input data
    index : int, optional
        Specific index to use. If None, picks a random example
    device : str, default='cuda'
        Device to run inference on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (input_image, activations, output, true_label)
        - input_image: numpy array of shape [1, 28, 28]
        - activations: numpy array of concatenated hidden layer activations
        - output: numpy array of shape [10]
        - true_label: int
    """
    # Get a batch from the loader
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # Select specific or random example
    idx = torch.randint(0, images.shape[0], (1,)).item() if index is None else index
    image = images[idx]
    label = labels[idx].item()
    
    activation_dict = {}
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for all hidden layers
    hooks = []
    
    # Add hooks for feature extraction layers (except first)
    feature_layers = list(model.features)[1:]  # Skip first layer
    for layer in feature_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Add hooks for embedding layers
    for layer in model.embedding:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Add hooks for classifier layers (except last)
    classifier_layers = list(model.classifier)[:-1]  # Skip last layer
    for layer in classifier_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        
        # Process hidden layer activations
        if activation_dict:
            activations_list = []
            for i in range(len(activation_dict)):
                act = activation_dict[i]
                # Flatten if needed
                if len(act.shape) > 2:
                    act = act.reshape(act.shape[0], -1)
                activations_list.append(act)
            
            # Concatenate all activations
            activations = np.concatenate(activations_list, axis=1)
            activations = activations[0]  # Remove batch dimension
        else:
            activations = np.array([])
        
        final_output = output.cpu().numpy()[0]
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return image.numpy(), activations, final_output, label

def collect_single_example_activations(model, loader, index=None, device='cuda'):
    """
    Collect hidden layer activations for a single example
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    loader : DataLoader
        DataLoader containing the input data
    index : int, optional
        Specific index to use. If None, picks a random example
    device : str, default='cuda'
        Device to run inference on ('cuda' or 'cpu')
        
    Returns:
    --------
    tuple
        (input_image, activations, output, true_label)
        - input_image: numpy array of shape [1, 28, 28]
        - activations: numpy array of concatenated hidden layer activations
        - output: numpy array of shape [10]
        - true_label: int
    """
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    idx = torch.randint(0, images.shape[0], (1,)).item() if index is None else index
    image = images[idx]
    label = labels[idx].item()
    
    activation_dict = {}
    def hook_fn(module, input, output):
        activation_dict[len(activation_dict)] = output.detach().cpu().numpy()
    
    # Register hooks for hidden layers only
    hooks = []
    layers = list(model.model)
    for layer in layers[1:-1]:  # Skip first and last layer
        hooks.append(layer.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        
        # Concatenate hidden layer activations
        if activation_dict:  # Only if we have hidden activations
            activations = np.concatenate([
                activation_dict[i].reshape(1, -1) 
                for i in range(len(activation_dict))
            ], axis=1)
            activations = activations[0]  # Remove batch dimension
        else:
            activations = np.array([])
        
        final_output = output.cpu().numpy()[0]
    
    for hook in hooks:
        hook.remove()
    
    return image.numpy(), activations, final_output, label

def knn_search(query, dataset, k=5, metric='euclidean'):
    """
    Perform KNN search for a query point in a dataset
    
    Parameters:
    -----------
    query : numpy.ndarray
        Query point of shape matching dataset samples
    dataset : numpy.ndarray
        Dataset to search in, shape [n_samples, n_features]
    k : int, default=5
        Number of neighbors to find
    metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cosine')
        
    Returns:
    --------
    indices : numpy.ndarray
        Indices of k nearest neighbors
    distances : numpy.ndarray
        Distances to k nearest neighbors
    """
    # Ensure inputs are numpy arrays with correct shape
    query = np.asarray(query).flatten()
    dataset = np.asarray(dataset)
    if len(dataset.shape) > 2:
        dataset = dataset.reshape(dataset.shape[0], -1)
    
    # Calculate distances
    if metric == 'euclidean':
        distances = np.sqrt(((dataset - query) ** 2).sum(axis=1))
    elif metric == 'cosine':
        norm_query = query / np.linalg.norm(query)
        norm_dataset = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
        distances = 1 - (norm_dataset @ norm_query)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get top-k indices and distances
    top_indices = np.argsort(distances)[:k]
    top_distances = distances[top_indices]
    
    return top_indices, top_distances

def find_optimal_k(query, dataset, k_range=(5, 50), step=5, metric='euclidean', reference=None):
    """
    Find optimal k for KNN search based on blending results
    
    Parameters:
    -----------
    query : numpy.ndarray
        Query point to find neighbors for
    dataset : numpy.ndarray
        Dataset to search in
    k_range : tuple, default=(5, 50)
        Range of k values to test (min_k, max_k)
    step : int, default=5
        Step size for k values
    metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'cosine')
    reference : numpy.ndarray, optional
        Reference sample to compare blended result against
        
    Returns:
    --------
    tuple
        (best_k, best_indices, best_distances, best_distance)
        - best_k: optimal k value found
        - best_indices: indices for best k
        - best_distances: distances for best k
        - best_distance: distance to reference for best k
    """
    min_k, max_k = k_range
    k_values = range(min_k, max_k + 1, step)
    best_k = min_k
    best_distance = float('inf')
    best_indices = None
    best_distances = None
    
    for k in k_values:
        # Get k nearest neighbors
        indices, distances = knn_search(
            query=query,
            dataset=dataset,
            k=k,
            metric=metric
        )
        
        # Blend samples and get distance
        blended, dist = blend_samples(
            dataset=dataset,
            indices=indices,
            distances=distances,
            reference=reference if reference is not None else query
        )
        
        # Update best if current distance is lower
        if dist < best_distance:
            best_distance = dist
            best_k = k
            best_indices = indices
            best_distances = distances
    
    return best_k, best_indices, best_distances, best_distance

def blend_samples(dataset, indices, distances, reference=None, epsilon=1e-10):
    """
    Blend samples using inverse distance weighting and optionally compare to reference
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Dataset containing samples to blend, shape [n_samples, ...]
    indices : numpy.ndarray
        Indices of samples to blend
    distances : numpy.ndarray
        Distances to selected samples, matching indices
    reference : numpy.ndarray, optional
        Reference sample to compare blended result against
    epsilon : float, default=1e-10
        Small constant to avoid division by zero
        
    Returns:
    --------
    blended : numpy.ndarray
        Blended sample using inverse distance weighting
    ref_distance : float or None
        L2 distance to reference sample if provided, else None
    """
    # Ensure inputs are numpy arrays
    dataset = np.asarray(dataset)
    distances = np.asarray(distances)
    
    # Calculate inverse distance weights
    weights = 1 / (distances + epsilon)
    weights = weights / np.sum(weights)
    
    # Select samples to blend
    samples = dataset[indices]
    
    # Reshape samples and weights for broadcasting
    samples_flat = samples.reshape(len(samples), -1)  # Flatten all dimensions except first
    weights = weights.reshape(-1, 1)  # Shape: [n_samples, 1]
    
    # Calculate weighted average
    blended_flat = np.sum(samples_flat * weights, axis=0)
    
    # Reshape back to original shape
    blended = blended_flat.reshape(dataset.shape[1:])
    
    # Calculate distance to reference if provided
    ref_distance = None
    if reference is not None:
        reference = np.asarray(reference).flatten()
        ref_distance = np.sqrt(np.sum((blended_flat - reference.flatten()) ** 2))
    
    return blended, ref_distance

def analyze_multiple_samples(model, loader, dataset_inputs, dataset_activations, dataset_outputs,
                           n_samples=50, k=15, device='cuda', seed=None,
                           reduced_activations=None, reduction_model=None, 
                           reduction_type=None, activation_range=None,
                           important_indices=None):
    """
    Analyze multiple samples using KNN search and blending with optional dimensionality reduction
    or important neuron selection
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    loader : DataLoader
        DataLoader containing input data
    dataset_inputs : numpy.ndarray
        Dataset of input samples
    dataset_activations : numpy.ndarray
        Dataset of activation samples
    dataset_outputs : numpy.ndarray
        Dataset of output samples
    n_samples : int, default=50
        Number of samples to analyze
    k : int, default=15
        Number of neighbors for KNN
    device : str, default='cuda'
        Device to run on
    seed : int, optional
        Random seed for reproducibility
    reduced_activations : numpy.ndarray, optional
        Pre-reduced activation dataset
    reduction_model : object, optional
        PCA or VAE model for reducing dimensions
    reduction_type : str, optional
        Type of reduction model ('pca' or 'vae')
    activation_range : tuple, optional
        Range of activation indices to use (start, end)
    important_indices : numpy.ndarray, optional
        Indices of important neurons to use for activation comparison
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    results = {
        'input_based': {'input_dist': [], 'output_dist': [], 'activation_dist': [], 'combined_dist': []},
        'activation_based': {'input_dist': [], 'output_dist': [], 'activation_dist': [], 'combined_dist': []},
        'output_based': {'input_dist': [], 'output_dist': [], 'activation_dist': [], 'combined_dist': []}
    }
    
    # Determine which activations to use for search
    if important_indices is not None:
        # Use only important neurons
        search_activations = dataset_activations[:, important_indices]
    elif reduced_activations is not None:
        # Use pre-reduced activations
        search_activations = reduced_activations
    else:
        # Use full or range-limited activations
        search_activations = dataset_activations
    
    # Collect all distances first
    all_distances = {
        'input_dist': [],
        'output_dist': [],
        'activation_dist': [],
        'combined_dist': []
    }
    
    # First pass to collect all distances
    for _ in range(n_samples):
        new_input, new_activations, new_output, _ = collect_single_example_activations(
            model, loader, device=device
        )
        
        # Process new activations based on configuration
        if important_indices is not None:
            new_search_activations = new_activations[important_indices]
        elif reduction_model is not None:
            if reduction_type == 'pca':
                new_search_activations = transform_new_activations(
                    new_activations=new_activations,
                    pca_model=reduction_model
                )
            elif reduction_type == 'vae':
                new_search_activations = transform_new_activations_vae(
                    new_activations=new_activations,
                    vae_model=reduction_model
                )
        else:
            new_search_activations = new_activations
            
        # Apply activation range if specified and not using important indices
        if activation_range is not None and important_indices is None:
            start, end = activation_range
            new_search_activations = new_search_activations[start:end]
            search_activations_subset = search_activations[:, start:end]
        else:
            search_activations_subset = search_activations
        
        # Collect distances for each approach
        for approach, query, dataset in [
            ('input_based', new_input, dataset_inputs),
            ('activation_based', new_search_activations, search_activations_subset),
            ('output_based', new_output, dataset_outputs)
        ]:
            indices, distances = knn_search(query=query, dataset=dataset, k=k)
            input_blend, in_dist = blend_samples(dataset_inputs, indices, distances, new_input)
            activation_blend, act_dist = blend_samples(dataset_activations, indices, distances, new_activations)
            output_blend, out_dist = blend_samples(dataset_outputs, indices, distances, new_output)
            
            all_distances['input_dist'].append(in_dist)
            all_distances['output_dist'].append(out_dist)
            all_distances['activation_dist'].append(act_dist)
            all_distances['combined_dist'].append(in_dist + out_dist + act_dist)
    
    # Calculate normalization parameters
    norm_params = {}
    for dist_type in ['input_dist', 'output_dist', 'activation_dist', 'combined_dist']:
        min_val = np.min(all_distances[dist_type])
        max_val = np.max(all_distances[dist_type])
        norm_params[dist_type] = (min_val, max_val)
    
    # Second pass with normalized distances
    for _ in range(n_samples):
        new_input, new_activations, new_output, _ = collect_single_example_activations(
            model, loader, device=device
        )
        
        # Process new activations based on configuration
        if important_indices is not None:
            new_search_activations = new_activations[important_indices]
        elif reduction_model is not None:
            if reduction_type == 'pca':
                new_search_activations = transform_new_activations(
                    new_activations=new_activations,
                    pca_model=reduction_model
                )
            elif reduction_type == 'vae':
                new_search_activations = transform_new_activations_vae(
                    new_activations=new_activations,
                    vae_model=reduction_model
                )
        else:
            new_search_activations = new_activations
            
        # Apply activation range if specified and not using important indices
        if activation_range is not None and important_indices is None:
            start, end = activation_range
            new_search_activations = new_search_activations[start:end]
            search_activations_subset = search_activations[:, start:end]
        else:
            search_activations_subset = search_activations
        
        # Process each approach
        for approach, query, dataset in [
            ('input_based', new_input, dataset_inputs),
            ('activation_based', new_search_activations, search_activations_subset),
            ('output_based', new_output, dataset_outputs)
        ]:
            indices, distances = knn_search(query=query, dataset=dataset, k=k)
            input_blend, in_dist = blend_samples(dataset_inputs, indices, distances, new_input)
            activation_blend, act_dist = blend_samples(dataset_activations, indices, distances, new_activations)
            output_blend, out_dist = blend_samples(dataset_outputs, indices, distances, new_output)
            
            # Normalize distances
            in_dist_norm = (in_dist - norm_params['input_dist'][0]) / (norm_params['input_dist'][1] - norm_params['input_dist'][0])
            act_dist_norm = (act_dist - norm_params['activation_dist'][0]) / (norm_params['activation_dist'][1] - norm_params['activation_dist'][0])
            out_dist_norm = (out_dist - norm_params['output_dist'][0]) / (norm_params['output_dist'][1] - norm_params['output_dist'][0])
            combined_dist_norm = (in_dist + out_dist + act_dist - norm_params['combined_dist'][0]) / (norm_params['combined_dist'][1] - norm_params['combined_dist'][0])
            
            # Store normalized distances
            results[approach]['input_dist'].append(in_dist_norm)
            results[approach]['activation_dist'].append(act_dist_norm)
            results[approach]['output_dist'].append(out_dist_norm)
            results[approach]['combined_dist'].append(combined_dist_norm)
    
    return results

def analyze_multiple_samples_embed(model, loader, dataset_inputs, dataset_activations, dataset_outputs,
                           n_samples=50, k=15, device='cuda', seed=None,
                           reduced_activations=None, reduction_model=None, 
                           reduction_type=None, activation_range=None,
                           important_indices=None):
    """
    Analyze multiple samples using KNN search and blending with optional dimensionality reduction
    or important neuron selection
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    loader : DataLoader
        DataLoader containing input data
    dataset_inputs : numpy.ndarray
        Dataset of input samples
    dataset_activations : numpy.ndarray
        Dataset of activation samples
    dataset_outputs : numpy.ndarray
        Dataset of output samples
    n_samples : int, default=50
        Number of samples to analyze
    k : int, default=15
        Number of neighbors for KNN
    device : str, default='cuda'
        Device to run on
    seed : int, optional
        Random seed for reproducibility
    reduced_activations : numpy.ndarray, optional
        Pre-reduced activation dataset
    reduction_model : object, optional
        PCA or VAE model for reducing dimensions
    reduction_type : str, optional
        Type of reduction model ('pca' or 'vae')
    activation_range : tuple, optional
        Range of activation indices to use (start, end)
    important_indices : numpy.ndarray, optional
        Indices of important neurons to use for activation comparison
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    results = {
        'input_based': {'input_dist': [], 'output_dist': [], 'activation_dist': [], 'combined_dist': []},
        'activation_based': {'input_dist': [], 'output_dist': [], 'activation_dist': [], 'combined_dist': []},
        'output_based': {'input_dist': [], 'output_dist': [], 'activation_dist': [], 'combined_dist': []}
    }
    
    # Determine which activations to use for search
    if important_indices is not None:
        # Use only important neurons
        search_activations = dataset_activations[:, important_indices]
    elif reduced_activations is not None:
        # Use pre-reduced activations
        search_activations = reduced_activations
    else:
        # Use full or range-limited activations
        search_activations = dataset_activations
    
    # Collect all distances first
    all_distances = {
        'input_dist': [],
        'output_dist': [],
        'activation_dist': [],
        'combined_dist': []
    }
    
    # First pass to collect all distances
    for _ in range(n_samples):
        new_input, new_activations, new_output, _ = collect_single_example_activations_embed(
            model, loader, device=device
        )
        
        # Process new activations based on configuration
        if important_indices is not None:
            new_search_activations = new_activations[important_indices]
        elif reduction_model is not None:
            if reduction_type == 'pca':
                new_search_activations = transform_new_activations(
                    new_activations=new_activations,
                    pca_model=reduction_model
                )
            elif reduction_type == 'vae':
                new_search_activations = transform_new_activations_vae(
                    new_activations=new_activations,
                    vae_model=reduction_model
                )
        else:
            new_search_activations = new_activations
            
        # Apply activation range if specified and not using important indices
        if activation_range is not None and important_indices is None:
            start, end = activation_range
            new_search_activations = new_search_activations[start:end]
            search_activations_subset = search_activations[:, start:end]
        else:
            search_activations_subset = search_activations
        
        # Collect distances for each approach
        for approach, query, dataset in [
            ('input_based', new_input, dataset_inputs),
            ('activation_based', new_search_activations, search_activations_subset),
            ('output_based', new_output, dataset_outputs)
        ]:
            indices, distances = knn_search(query=query, dataset=dataset, k=k)
            input_blend, in_dist = blend_samples(dataset_inputs, indices, distances, new_input)
            activation_blend, act_dist = blend_samples(dataset_activations, indices, distances, new_activations)
            output_blend, out_dist = blend_samples(dataset_outputs, indices, distances, new_output)
            
            all_distances['input_dist'].append(in_dist)
            all_distances['output_dist'].append(out_dist)
            all_distances['activation_dist'].append(act_dist)
            all_distances['combined_dist'].append(in_dist + out_dist + act_dist)
    
    # Calculate normalization parameters
    norm_params = {}
    for dist_type in ['input_dist', 'output_dist', 'activation_dist', 'combined_dist']:
        min_val = np.min(all_distances[dist_type])
        max_val = np.max(all_distances[dist_type])
        norm_params[dist_type] = (min_val, max_val)
    
    # Second pass with normalized distances
    for _ in range(n_samples):
        new_input, new_activations, new_output, _ = collect_single_example_activations_embed(
            model, loader, device=device
        )
        
        # Process new activations based on configuration
        if important_indices is not None:
            new_search_activations = new_activations[important_indices]
        elif reduction_model is not None:
            if reduction_type == 'pca':
                new_search_activations = transform_new_activations(
                    new_activations=new_activations,
                    pca_model=reduction_model
                )
            elif reduction_type == 'vae':
                new_search_activations = transform_new_activations_vae(
                    new_activations=new_activations,
                    vae_model=reduction_model
                )
        else:
            new_search_activations = new_activations
            
        # Apply activation range if specified and not using important indices
        if activation_range is not None and important_indices is None:
            start, end = activation_range
            new_search_activations = new_search_activations[start:end]
            search_activations_subset = search_activations[:, start:end]
        else:
            search_activations_subset = search_activations
        
        # Process each approach
        for approach, query, dataset in [
            ('input_based', new_input, dataset_inputs),
            ('activation_based', new_search_activations, search_activations_subset),
            ('output_based', new_output, dataset_outputs)
        ]:
            indices, distances = knn_search(query=query, dataset=dataset, k=k)
            input_blend, in_dist = blend_samples(dataset_inputs, indices, distances, new_input)
            activation_blend, act_dist = blend_samples(dataset_activations, indices, distances, new_activations)
            output_blend, out_dist = blend_samples(dataset_outputs, indices, distances, new_output)
            
            # Normalize distances
            in_dist_norm = (in_dist - norm_params['input_dist'][0]) / (norm_params['input_dist'][1] - norm_params['input_dist'][0])
            act_dist_norm = (act_dist - norm_params['activation_dist'][0]) / (norm_params['activation_dist'][1] - norm_params['activation_dist'][0])
            out_dist_norm = (out_dist - norm_params['output_dist'][0]) / (norm_params['output_dist'][1] - norm_params['output_dist'][0])
            combined_dist_norm = (in_dist + out_dist + act_dist - norm_params['combined_dist'][0]) / (norm_params['combined_dist'][1] - norm_params['combined_dist'][0])
            
            # Store normalized distances
            results[approach]['input_dist'].append(in_dist_norm)
            results[approach]['activation_dist'].append(act_dist_norm)
            results[approach]['output_dist'].append(out_dist_norm)
            results[approach]['combined_dist'].append(combined_dist_norm)
    
    return results

def analyze_multiple_samples_adaptive(model, loader, dataset_inputs, dataset_activations, dataset_outputs, 
                                   n_samples=50, k_range=(5,50), step=5, device='cuda', 
                                   activation_range=None, seed=None):
    """
    Analyze multiple samples using adaptive KNN search with activation distance blending checks
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize results dictionary
    results = {
        'input_based': {
            'input_dist': [], 'output_dist': [], 'activation_dist': [], 
            'blended_activation_dist': [], 'combined_dist': [], 'optimal_k': []
        },
        'activation_based': {
            'input_dist': [], 'output_dist': [], 'activation_dist': [], 
            'blended_activation_dist': [], 'combined_dist': [], 'optimal_k': []
        },
        'output_based': {
            'input_dist': [], 'output_dist': [], 'activation_dist': [], 
            'blended_activation_dist': [], 'combined_dist': [], 'optimal_k': []
        }
    }
    
    # Apply activation range if specified
    if activation_range is not None:
        start, end = activation_range
        dataset_activations = dataset_activations[:, start:end]
    
    # First pass to collect all distances for normalization
    all_distances = {
        'input_dist': [],
        'output_dist': [],
        'activation_dist': [],
        'blended_activation_dist': [],
        'combined_dist': []
    }
    
    # Collect distances for normalization
    for _ in range(n_samples):
        new_input, new_activations, new_output, _ = collect_single_example_activations(
            model, loader, device=device
        )
        
        if activation_range is not None:
            start, end = activation_range
            new_activations = new_activations[start:end]
            
        for query, dataset in [
            (new_input, dataset_inputs),
            (new_activations, dataset_activations),
            (new_output, dataset_outputs)
        ]:
            best_k, indices, distances, _ = find_optimal_k(
                query=query,
                dataset=dataset,
                k_range=k_range,
                step=step,
                reference=query
            )
            
            input_blend, in_dist = blend_samples(dataset_inputs, indices, distances, new_input)
            activation_blend, act_dist = blend_samples(dataset_activations, indices, distances, new_activations)
            output_blend, out_dist = blend_samples(dataset_outputs, indices, distances, new_output)
            
            # Get blended activation distance
            _, blended_act_dist = blend_samples(
                dataset_activations, 
                indices, 
                distances,
                activation_blend  # Compare to blended activations
            )
            
            all_distances['input_dist'].append(in_dist)
            all_distances['output_dist'].append(out_dist)
            all_distances['activation_dist'].append(act_dist)
            all_distances['blended_activation_dist'].append(blended_act_dist)
            all_distances['combined_dist'].append(in_dist + out_dist + act_dist + blended_act_dist)
    
    # Calculate normalization parameters
    norm_params = {}
    for dist_type in ['input_dist', 'output_dist', 'activation_dist', 
                     'blended_activation_dist', 'combined_dist']:
        min_val = np.min(all_distances[dist_type])
        max_val = np.max(all_distances[dist_type])
        norm_params[dist_type] = (min_val, max_val)
    
    # Second pass with normalized distances
    for _ in range(n_samples):
        new_input, new_activations, new_output, _ = collect_single_example_activations(
            model, loader, device=device
        )
        
        if activation_range is not None:
            start, end = activation_range
            new_activations = new_activations[start:end]
        
        # Process each approach (input-based, activation-based, output-based)
        for approach, query, dataset in [
            ('input_based', new_input, dataset_inputs),
            ('activation_based', new_activations, dataset_activations),
            ('output_based', new_output, dataset_outputs)
        ]:
            # Find optimal k
            best_k, indices, distances, _ = find_optimal_k(
                query=query,
                dataset=dataset,
                k_range=k_range,
                step=step,
                reference=query
            )
            
            # Calculate all distances
            input_blend, in_dist = blend_samples(dataset_inputs, indices, distances, new_input)
            activation_blend, act_dist = blend_samples(dataset_activations, indices, distances, new_activations)
            output_blend, out_dist = blend_samples(dataset_outputs, indices, distances, new_output)
            
            # Get blended activation distance
            _, blended_act_dist = blend_samples(
                dataset_activations, 
                indices, 
                distances,
                activation_blend
            )
            
            # Normalize distances
            in_dist_norm = (in_dist - norm_params['input_dist'][0]) / (norm_params['input_dist'][1] - norm_params['input_dist'][0])
            act_dist_norm = (act_dist - norm_params['activation_dist'][0]) / (norm_params['activation_dist'][1] - norm_params['activation_dist'][0])
            out_dist_norm = (out_dist - norm_params['output_dist'][0]) / (norm_params['output_dist'][1] - norm_params['output_dist'][0])
            blended_act_dist_norm = (blended_act_dist - norm_params['blended_activation_dist'][0]) / (norm_params['blended_activation_dist'][1] - norm_params['blended_activation_dist'][0])
            combined_dist_norm = (in_dist + out_dist + act_dist + blended_act_dist - norm_params['combined_dist'][0]) / (norm_params['combined_dist'][1] - norm_params['combined_dist'][0])
            
            # Store results
            results[approach]['input_dist'].append(in_dist_norm)
            results[approach]['activation_dist'].append(act_dist_norm)
            results[approach]['output_dist'].append(out_dist_norm)
            results[approach]['blended_activation_dist'].append(blended_act_dist_norm)
            results[approach]['combined_dist'].append(combined_dist_norm)
            results[approach]['optimal_k'].append(best_k)
    
    return results

def plot_distance_distributions(results):
    """
    Plot boxplots of distances for different KNN approaches
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_multiple_samples
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare data for plotting
    methods = ['input_based', 'activation_based', 'output_based']
    
    # Plot input distances
    input_data = [results[m]['input_dist'] for m in methods]
    ax1.boxplot(input_data, labels=['Input', 'Activation', 'Output'])
    ax1.set_title('Input Space Distances')
    ax1.set_ylabel('Normalized Distance')
    ax1.grid(True, alpha=0.3)
    
    # Plot activation distances
    activation_data = [results[m]['activation_dist'] for m in methods]
    ax2.boxplot(activation_data, labels=['Input', 'Activation', 'Output'])
    ax2.set_title('Activation Space Distances')
    ax2.set_ylabel('Normalized Distance')
    ax2.grid(True, alpha=0.3)
    
    # Plot output distances
    output_data = [results[m]['output_dist'] for m in methods]
    ax3.boxplot(output_data, labels=['Input', 'Activation', 'Output'])
    ax3.set_title('Output Space Distances')
    ax3.set_ylabel('Normalized Distance')
    ax3.grid(True, alpha=0.3)
    
    # Plot combined distances
    combined_data = [results[m]['combined_dist'] for m in methods]
    ax4.boxplot(combined_data, labels=['Input', 'Activation', 'Output'])
    ax4.set_title('Combined Distances')
    ax4.set_ylabel('Normalized Distance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_distance_distributions_adaptive(results):
    """Plot boxplots of distances and optimal k values for different KNN approaches"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
    
    methods = ['input_based', 'activation_based', 'output_based']
    
    # Plot input distances
    input_data = [results[m]['input_dist'] for m in methods]
    ax1.boxplot(input_data, labels=['Input', 'Activation', 'Output'])
    ax1.set_title('Input Space Distances')
    ax1.set_ylabel('Normalized Distance')
    ax1.grid(True, alpha=0.3)
    
    # Plot activation distances
    activation_data = [results[m]['activation_dist'] for m in methods]
    ax2.boxplot(activation_data, labels=['Input', 'Activation', 'Output'])
    ax2.set_title('Activation Space Distances')
    ax2.set_ylabel('Normalized Distance')
    ax2.grid(True, alpha=0.3)
    
    # Plot blended activation distances
    blended_act_data = [results[m]['blended_activation_dist'] for m in methods]
    ax3.boxplot(blended_act_data, labels=['Input', 'Activation', 'Output'])
    ax3.set_title('Blended Activation Distances')
    ax3.set_ylabel('Normalized Distance')
    ax3.grid(True, alpha=0.3)
    
    # Plot output distances
    output_data = [results[m]['output_dist'] for m in methods]
    ax4.boxplot(output_data, labels=['Input', 'Activation', 'Output'])
    ax4.set_title('Output Space Distances')
    ax4.set_ylabel('Normalized Distance')
    ax4.grid(True, alpha=0.3)
    
    # Plot combined distances
    combined_data = [results[m]['combined_dist'] for m in methods]
    ax5.boxplot(combined_data, labels=['Input', 'Activation', 'Output'])
    ax5.set_title('Combined Distances')
    ax5.set_ylabel('Normalized Distance')
    ax5.grid(True, alpha=0.3)
    
    # Plot optimal k values
    k_data = [results[m]['optimal_k'] for m in methods]
    ax6.boxplot(k_data, labels=['Input', 'Activation', 'Output'])
    ax6.set_title('Optimal k Values')
    ax6.set_ylabel('k')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print mean optimal k values and distances
    print("\nMean optimal k values:")
    for method in methods:
        mean_k = np.mean(results[method]['optimal_k'])
        std_k = np.std(results[method]['optimal_k'])
        print(f"{method}: {mean_k:.1f} ± {std_k:.1f}")
    
    print("\nMean distances for each approach:")
    for method in methods:
        print(f"\n{method}:")
        print(f"Input distance: {np.mean(results[method]['input_dist']):.4f}")
        print(f"Activation distance: {np.mean(results[method]['activation_dist']):.4f}")
        print(f"Blended activation distance: {np.mean(results[method]['blended_activation_dist']):.4f}")
        print(f"Output distance: {np.mean(results[method]['output_dist']):.4f}")
        print(f"Combined distance: {np.mean(results[method]['combined_dist']):.4f}")

def evaluate_per_class(model, test_loader, device='cuda'):
    """
    Evaluate model accuracy for each MNIST digit class
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model to evaluate
    test_loader : DataLoader
        DataLoader containing test data
    device : str, default='cuda'
        Device to run evaluation on
        
    Returns:
    --------
    dict
        Dictionary containing per-class accuracies
    """
    model.eval()
    # Initialize counters for each class
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update counters for each class
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
    
    # Calculate and print accuracy for each class
    print("\nAccuracy per class:")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Digit {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return {i: 100 * class_correct[i] / class_total[i] for i in range(10)}

def find_activation_extremes(activations, activation_range=None, n_neurons=10, get_highest=True):
    """
    Find neurons with highest/lowest average activation in specified range
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Activation dataset of shape [n_samples, n_features]
    activation_range : tuple, optional
        Tuple of (start, end) indices to analyze specific activation range
        If None, analyzes all activations
    n_neurons : int, default=10
        Number of neurons to return
    get_highest : bool, default=True
        If True, returns highest activating neurons
        If False, returns lowest activating neurons
        
    Returns:
    --------
    tuple
        (indices, mean_activations)
        - indices: indices of selected neurons relative to original activation array
        - mean_activations: average activation values for selected neurons
    """
    # Apply activation range if specified
    if activation_range is not None:
        start, end = activation_range
        activations_subset = activations[:, start:end]
        offset = start
    else:
        activations_subset = activations
        offset = 0
    
    # Calculate mean activation per neuron
    mean_activations = np.mean(activations_subset, axis=0)
    
    # Get indices of highest/lowest activating neurons
    if get_highest:
        indices = np.argsort(mean_activations)[-n_neurons:][::-1]  # Highest n neurons
    else:
        indices = np.argsort(mean_activations)[:n_neurons]  # Lowest n neurons
    
    # Adjust indices to match original activation array
    original_indices = indices + offset
    selected_means = mean_activations[indices]
    
    return original_indices, selected_means

def find_activation_extremes_per_class(activations, labels, activation_range=None, n_neurons=10, get_highest=True):
    """
    Find neurons with highest/lowest average activation for each class
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Activation dataset of shape [n_samples, n_features]
    labels : numpy.ndarray
        Class labels for each sample
    activation_range : tuple, optional
        Tuple of (start, end) indices to analyze specific activation range
    n_neurons : int, default=10
        Number of neurons to return per class
    get_highest : bool, default=True
        If True, returns highest activating neurons
        If False, returns lowest activating neurons
        
    Returns:
    --------
    dict
        Dictionary containing per-class results with:
        - indices: indices of selected neurons
        - means: mean activation values
    """
    # Apply activation range if specified
    if activation_range is not None:
        start, end = activation_range
        activations_subset = activations[:, start:end]
        offset = start
    else:
        activations_subset = activations
        offset = 0
    
    results = {}
    
    # Process each class separately
    for class_idx in range(10):  # Assuming 10 classes for MNIST
        # Get activations for current class
        class_mask = labels == class_idx
        class_activations = activations_subset[class_mask]
        
        # Calculate mean activation per neuron for this class
        class_means = np.mean(class_activations, axis=0)
        
        # Get indices of highest/lowest activating neurons
        if get_highest:
            indices = np.argsort(class_means)[-n_neurons:][::-1]
        else:
            indices = np.argsort(class_means)[:n_neurons]
        
        # Adjust indices to match original activation array
        original_indices = indices + offset
        selected_means = class_means[indices]
        
        # Store results for this class
        results[class_idx] = {
            'indices': original_indices,
            'means': selected_means
        }
    
    return results


def plot_activation_extremes_per_class(results, get_highest=True):
    """
    Visualize activation patterns for each class
    
    Parameters:
    -----------
    results : dict
        Results dictionary from find_activation_extremes_per_class
    get_highest : bool, default=True
        Whether these are highest or lowest activations
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    type_str = "Highest" if get_highest else "Lowest"
    
    for class_idx in range(10):
        means = results[class_idx]['means']
        ax = axes[class_idx]
        
        ax.bar(range(len(means)), means)
        ax.set_title(f'{type_str} Activations\nClass {class_idx}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Mean Activation')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_per_class(model, test_loader, device='cuda'):
    """
    Evaluate model accuracy for each MNIST digit class
    """
    # Make sure model is on the correct device
    model = model.to(device)
    model.eval()
    
    # Initialize counters for each class
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move tensors to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update counters for each class
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                class_total[label_item] += 1
                if label_item == pred.item():
                    class_correct[label_item] += 1
    
    # Calculate accuracies
    accuracies = {}
    print("\nAccuracy per class:")
    for i in range(10):
        if class_total[i] > 0:  # Avoid division by zero
            accuracy = 100.0 * class_correct[i] / class_total[i]
            accuracies[i] = accuracy
            print(f"Digit {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            accuracies[i] = 0.0
            print(f"Digit {i}: No samples")
    
    return accuracies



def create_pca_model(activations, n_components=64, standardize=True):
    """
    Create and fit a PCA model to reduce activation dimensions
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Input activations of shape [n_samples, n_features]
    n_components : int, default=64
        Number of PCA components to keep
    standardize : bool, default=True
        Whether to standardize the data before PCA
    
    Returns:
    --------
    tuple
        (reduced_activations, pca_model, scaler)
        - reduced_activations: PCA-transformed activations
        - pca_model: Fitted PCA model for later use
        - scaler: Fitted StandardScaler (None if standardize=False)
    """
    # Initialize models
    scaler = StandardScaler() if standardize else None
    pca = PCA(n_components=n_components)
    
    # Standardize if requested
    if standardize:
        activations_scaled = scaler.fit_transform(activations)
    else:
        activations_scaled = activations
    
    # Fit PCA and transform data
    reduced_activations = pca.fit_transform(activations_scaled)
    
    # Print variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Explained variance with {n_components} components: {cumulative_variance[-1]:.4f}")
    
    return reduced_activations, pca, scaler

def transform_new_activations(new_activations, pca_model, scaler=None):
    """
    Transform new activations using pre-fitted PCA model
    
    Parameters:
    -----------
    new_activations : numpy.ndarray
        New activations to transform
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model from create_pca_model
    scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted StandardScaler from create_pca_model
    
    Returns:
    --------
    numpy.ndarray
        PCA-transformed activations
    """
    # Ensure input is 2D
    if new_activations.ndim == 1:
        new_activations = new_activations.reshape(1, -1)
    
    # Scale if scaler is provided
    if scaler is not None:
        new_activations = scaler.transform(new_activations)
    
    # Transform using PCA
    return pca_model.transform(new_activations)

def plot_cumulative_variance(pca_model):
    """
    Plot cumulative explained variance ratio
    
    Parameters:
    -----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    """
    cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines at common thresholds
    thresholds = [0.8, 0.9, 0.95]
    for threshold in thresholds:
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
        # Find number of components needed for this threshold
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        plt.text(len(cumulative_variance) * 0.6, threshold, 
                f'{threshold:.0%}: {n_components} components',
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

    # VAE Stuff



class ActivationVAE(nn.Module):
    """VAE for reducing activation dimensions with improved architecture"""
    def __init__(self, input_dim, latent_dim=64, hidden_dims=None):
        super(ActivationVAE, self).__init__()
        
        # Default hidden dimensions if not specified
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        # Build encoder layers
        encoder_layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ])
            last_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder layers
        decoder_layers = []
        hidden_dims.reverse()
        last_dim = latent_dim
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ])
            last_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def create_vae_model(activations, latent_dim=64, batch_size=128, num_epochs=50, 
                    learning_rate=1e-3, device='cuda', beta=1.0):
    """
    Create and train VAE model with improved training process
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Input activations
    latent_dim : int
        Size of latent dimension
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    device : str
        Device to train on
    beta : float
        Weight for KL divergence term (β-VAE)
    """
    # Data preparation
    activations_tensor = torch.FloatTensor(activations)
    dataset = TensorDataset(activations_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = activations.shape[1]
    hidden_dims = [min(input_dim, 512), 256, 128]  # Adaptive hidden dimensions
    vae = ActivationVAE(input_dim, latent_dim, hidden_dims).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    vae.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            
            # Forward pass
            recon_x, mu, log_var = vae(x)
            
            # Loss calculation
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + beta * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Generate reduced activations
    vae.eval()
    reduced_activations = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            mu, _ = vae.encode(x)
            reduced_activations.append(mu.cpu().numpy())
    
    reduced_activations = np.concatenate(reduced_activations, axis=0)
    return reduced_activations, vae

def transform_new_activations_vae(new_activations, vae_model, device='cuda'):
    """
    Transform new activations using trained VAE model
    
    Parameters:
    -----------
    new_activations : numpy.ndarray
        New activations to transform
    vae_model : ActivationVAE
        Trained VAE model from create_vae_model
    device : str, default='cuda'
        Device to run inference on
        
    Returns:
    --------
    numpy.ndarray
        VAE-encoded activations
    """
    # Ensure input is 2D
    if new_activations.ndim == 1:
        new_activations = new_activations.reshape(1, -1)
    
    # Convert to torch tensor
    new_activations = torch.FloatTensor(new_activations).to(device)
    
    # Encode
    vae_model.eval()
    with torch.no_grad():
        mu, _ = vae_model.encode(new_activations)
    
    return mu.cpu().numpy()

def find_important_neurons_in_range(activations, start_idx, end_idx, n_neurons=None):
    """
    Find neuron importance scores based on variance and activity
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Array of shape [n_samples, n_features] containing activation values
    start_idx : int
        Start index of the range to analyze
    end_idx : int
        End index of the range to analyze
    n_neurons : int, optional
        Number of neurons to return. If None, returns scores for all neurons
        
    Returns:
    --------
    numpy.ndarray
        Indices of neurons sorted by importance (most to least important)
    numpy.ndarray
        Importance scores for all neurons
    """
    # Extract the range we want to analyze
    activations_subset = activations[:, start_idx:end_idx]
    
    # Calculate variance (activity spread)
    variance_scores = np.var(activations_subset, axis=0)
    
    # Calculate mean absolute activation (activity magnitude)
    magnitude_scores = np.mean(np.abs(activations_subset), axis=0)
    
    # Normalize scores
    variance_norm = (variance_scores - variance_scores.min()) / (variance_scores.max() - variance_scores.min() + 1e-10)
    magnitude_norm = (magnitude_scores - magnitude_scores.min()) / (magnitude_scores.max() - magnitude_scores.min() + 1e-10)
    
    # Combine scores
    importance_scores = variance_norm + magnitude_norm
    
    # Sort neurons by importance (highest to lowest)
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    if n_neurons is not None:
        sorted_indices = sorted_indices[:n_neurons]
    
    # Adjust indices to match original activation array
    original_indices = sorted_indices + start_idx
    
    return original_indices, importance_scores

def calculate_accuracy(model, loader, device='cuda'):
    """
    Calculate overall accuracy of the model on given data loader
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to evaluate
    loader : DataLoader
        DataLoader containing validation or test data
    device : str, default='cuda'
        Device to run evaluation on
        
    Returns:
    --------
    float
        Accuracy as percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def create_pruned_model(original_model, indices_to_zero, device='cuda'):
    """
    Create a copy of the model and zero out weights for specified neurons
    
    Parameters:
    -----------
    original_model : torch.nn.Module
        The original neural network model
    indices_to_zero : numpy.ndarray or list
        Indices of neurons to zero out
    device : str, default='cuda'
        Device to place the model on
        
    Returns:
    --------
    torch.nn.Module
        New model with zeroed weights at specified indices
    """
    # Create a deep copy of the model
    pruned_model = copy.deepcopy(original_model)
    pruned_model = pruned_model.to(device)
    
    # Set model to eval mode
    pruned_model.eval()
    
    # Get the layers of the model
    layers = list(pruned_model.model)
    
    # Find all Linear layers
    linear_layers = [(i, layer) for i, layer in enumerate(layers) 
                    if isinstance(layer, nn.Linear)]
    
    # For each specified index, zero out the corresponding weights and bias
    for idx in indices_to_zero:
        # Calculate which layer this neuron belongs to
        current_pos = 0
        for layer_idx, layer in linear_layers:
            output_size = layer.out_features
            
            # Check if the index falls within this layer's outputs
            if current_pos <= idx < current_pos + output_size:
                # Calculate the relative position within this layer
                relative_idx = idx - current_pos
                
                # Zero out weights and bias for this neuron
                layer.weight.data[relative_idx].zero_()
                if layer.bias is not None:
                    layer.bias.data[relative_idx] = 0
                break
                
            current_pos += output_size
    
    return pruned_model

def split_activations_by_class(activations, labels):
    """
    Split activation data by class label
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Array of activations with shape [n_samples, n_features]
    labels : numpy.ndarray
        Array of class labels with shape [n_samples]
        
    Returns:
    --------
    dict
        Dictionary containing activations for each class
        {class_label: class_activations_array}
    """
    # Initialize dictionary to store class-specific activations
    class_activations = {}
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Split activations by class
    for class_label in unique_classes:
        # Create mask for current class
        class_mask = labels == class_label
        # Store activations for current class
        class_activations[class_label] = activations[class_mask]
    
    return class_activations

def evaluate_per_class(model, test_loader, device='cuda'):
    """
    Evaluate model accuracy for each MNIST digit class
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model to evaluate
    test_loader : DataLoader
        DataLoader containing test data
    device : str, default='cuda'
        Device to run evaluation on
        
    Returns:
    --------
    dict
        Dictionary containing per-class accuracies
    """
    model.eval()
    # Initialize counters for each class
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update counters for each class
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
    
    # Calculate and print accuracy for each class
    print("\nAccuracy per class:")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Digit {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return {i: 100 * class_correct[i] / class_total[i] for i in range(10)}

def evaluate_pruning_strategies(model, test_loader, activations, num_steps=20, max_neurons=256, device='cuda', seed=42):
    """
    Compare pruning based on importance vs random selection
    """
    np.random.seed(seed)
    total_neurons = activations.shape[1]
    neurons_per_step = max_neurons // num_steps
    
    results = {
        'importance_based': {'neurons': [], 'accuracy': []},
        'random': {'neurons': [], 'accuracy': []}
    }
    
    # Get importance-based ordering of all neurons
    indices, scores = find_important_neurons_in_range(
        activations=activations,
        start_idx=0,
        end_idx=total_neurons
    )
    
    # Indices are already sorted from most to least important
    importance_order = indices
    
    # Create random ordering
    random_order = np.random.permutation(total_neurons)
    
    # Store baseline accuracy
    baseline_acc = calculate_accuracy(model, test_loader, device)
    results['importance_based']['neurons'].append(0)
    results['importance_based']['accuracy'].append(baseline_acc)
    results['random']['neurons'].append(0)
    results['random']['accuracy'].append(baseline_acc)
    
    # Evaluate progressively larger sets of pruned neurons
    for i in range(num_steps):
        n_neurons = (i + 1) * neurons_per_step
        
        # Importance-based pruning (prune least important neurons)
        neurons_to_prune = importance_order[-n_neurons:]
        pruned_model = create_pruned_model(model, neurons_to_prune, device)
        importance_acc = calculate_accuracy(pruned_model, test_loader, device)
        
        # Random pruning
        random_neurons = random_order[:n_neurons]
        random_pruned_model = create_pruned_model(model, random_neurons, device)
        random_acc = calculate_accuracy(random_pruned_model, test_loader, device)
        
        # Store results
        results['importance_based']['neurons'].append(n_neurons)
        results['importance_based']['accuracy'].append(importance_acc)
        results['random']['neurons'].append(n_neurons)
        results['random']['accuracy'].append(random_acc)
        
        print(f"Pruned {n_neurons} neurons:")
        print(f"  Importance-based accuracy: {importance_acc:.2f}%")
        print(f"  Random pruning accuracy: {random_acc:.2f}%")
    
    return results

def plot_pruning_comparison(results):
    """
    Plot comparison of pruning strategies
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_pruning_strategies
    """
    plt.figure(figsize=(10, 6))
    
    # Plot both strategies
    plt.plot(results['importance_based']['neurons'], 
             results['importance_based']['accuracy'], 
             'b-', label='Importance-based pruning')
    plt.plot(results['random']['neurons'], 
             results['random']['accuracy'], 
             'r--', label='Random pruning')
    
    plt.xlabel('Number of Pruned Neurons')
    plt.ylabel('Model Accuracy (%)')
    plt.title('Comparison of Pruning Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines at specific accuracy drops
    baseline = results['importance_based']['accuracy'][0]
    thresholds = [0.99, 0.95, 0.90]
    colors = ['g', 'y', 'r']
    
    for threshold, color in zip(thresholds, colors):
        threshold_acc = baseline * threshold
        
        # Find crossing points for both strategies
        for strategy in ['importance_based', 'random']:
            accuracies = np.array(results[strategy]['accuracy'])
            neurons = np.array(results[strategy]['neurons'])
            
            # Find where accuracy drops below threshold
            if any(accuracies < threshold_acc):
                idx = np.where(accuracies < threshold_acc)[0][0]
                plt.axvline(x=neurons[idx], color=color, alpha=0.3, linestyle=':')
                plt.text(neurons[idx], plt.ylim()[0], 
                        f'{int(threshold*100)}%\n{neurons[idx]}n',
                        rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

def compare_pruning_strategies(model, test_loader, important_indices, device='cuda'):
    """
    Compare pruning using important neurons vs random neurons
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    test_loader : DataLoader
        Test data loader
    important_indices : list or numpy.ndarray
        List of neuron indices sorted by importance (most to least important)
    device : str, default='cuda'
        Device to run evaluation on
    """
    n_neurons = len(important_indices)
    
    # Create random indices
    random_indices = np.random.randint(0, 512, size=len(important_indices))

    
    # Initialize results
    results = {
        'n_pruned': [0],  # Start with 0 pruned neurons
        'importance_acc': [calculate_accuracy(model, test_loader, device)],  # Baseline accuracy
        'random_acc': [calculate_accuracy(model, test_loader, device)]  # Baseline accuracy
    }
    
    # Evaluate progressively larger sets of pruned neurons
    for i in range(n_neurons):
        # Importance-based pruning (prune least important first)
        importance_neurons = important_indices[-(i+1):]  # Take last n neurons
        pruned_model = create_pruned_model(model, importance_neurons, device)
        importance_acc = calculate_accuracy(pruned_model, test_loader, device)
        
        # Random pruning
        random_neurons = random_indices[:i+1]  # Take first n neurons
        random_model = create_pruned_model(model, random_neurons, device)
        random_acc = calculate_accuracy(random_model, test_loader, device)
        
        # Store results
        results['n_pruned'].append(i+1)
        results['importance_acc'].append(importance_acc)
        results['random_acc'].append(random_acc)
        
        print(f"Pruned {i+1}/{n_neurons} neurons:")
        print(f"  Importance-based accuracy: {importance_acc:.2f}%")
        print(f"  Random pruning accuracy: {random_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['n_pruned'], results['importance_acc'], 'b-', label='Importance-based pruning')
    plt.plot(results['n_pruned'], results['random_acc'], 'r--', label='Random pruning')
    plt.xlabel('Number of Pruned Neurons')
    plt.ylabel('Model Accuracy (%)')
    plt.title('Comparison of Pruning Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

def compare_activation_distances(results_full=None, results_first=None, results_last=None, normalize=True):
    """
    Compare activation-based distances across network sections and optionally full network
    
    Parameters:
    -----------
    results_conv : dict, optional
        Results from analyze_multiple_samples for convolution layers
    results_embedding : dict, optional
        Results from analyze_multiple_samples for embedding layer
    results_classification : dict, optional
        Results from analyze_multiple_samples for classification layers
    results_full : dict, optional
        Results from analyze_multiple_samples for full network
    normalize : bool, default=True
        Whether to normalize distances to [0,1] range across all distances
    """
    # Get all distances for activation-based approach
    distances = {
        'Input Distance': {},
        'Activation Distance': {},
        'Output Distance': {},
        'Combined Distance': {}
    }
    
    # Add section results if provided
    sections = []
    if results_full:
        sections.append(('full', results_full))
    if results_first:
        sections.append(('first', results_first))
    if results_last:
        sections.append(('last', results_last))
    
    
    # Collect distances
    for dist_type in distances:
        for section_name, results in sections:
            if dist_type == 'Input Distance':
                distances[dist_type][section_name] = np.mean(results['activation_based']['input_dist'])
            elif dist_type == 'Activation Distance':
                distances[dist_type][section_name] = np.mean(results['activation_based']['activation_dist'])
            elif dist_type == 'Output Distance':
                distances[dist_type][section_name] = np.mean(results['activation_based']['output_dist'])
            else:  # Combined Distance
                distances[dist_type][section_name] = np.mean(results['activation_based']['combined_dist'])
    
    # Normalize if requested
    if normalize:
        all_values = [v for d in distances.values() for v in d.values()]
        min_val = min(all_values)
        max_val = max(all_values)
        for dist_type in distances:
            for section in distances[dist_type]:
                distances[dist_type][section] = (distances[dist_type][section] - min_val) / (max_val - min_val)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sections))  # Number of sections
    width = 0.2  # Width of bars
    multiplier = 0
    
    # Plot each distance type
    for dist_type, dist_values in distances.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, list(dist_values.values()), width, label=dist_type)
        
        # Add value labels on bars
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', rotation=0)
        
        multiplier += 1
    
    # Customize plot
    ax.set_ylabel('Normalized Distance' if normalize else 'Mean Distance')
    ax.set_title('Activation-Based Distances Across Network Sections')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([section[0] for section in sections])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print numeric values
    print("\nDetailed distances (activation-based approach):")
    for dist_type in distances:
        print(f"\n{dist_type}:")
        for section, value in distances[dist_type].items():
            print(f"{section}: {value:.3f}")

def visualize_network_structure(model, input_size=(1, 1, 28, 28)):
    """
    Visualize the network structure by showing tensor shapes at each layer
    """
    def hook_fn(module, input, output):
        activation_shapes[module] = output.shape
    
    activation_shapes = {}
    hooks = []
    
    # Register hooks for all layers
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.Flatten)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass with dummy input
    # Ensure input tensor is on the same device as model
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    
    model.eval()
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print network structure
    print("Network Structure:")
    print("-" * 50)
    
    # Feature extraction part
    print("\nFeature Extraction:")
    for layer in model.features:
        if layer in activation_shapes:
            print(f"{layer.__class__.__name__:15} Output shape: {activation_shapes[layer]}")
    
    # Embedding part
    print("\nEmbedding Layer:")
    for layer in model.embedding:
        if layer in activation_shapes:
            print(f"{layer.__class__.__name__:15} Output shape: {activation_shapes[layer]}")
    
    # Classification part
    print("\nClassification Head:")
    for layer in model.classifier:
        if layer in activation_shapes:
            print(f"{layer.__class__.__name__:15} Output shape: {activation_shapes[layer]}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def visualize_layer_sizes(model, input_size=(1, 1, 28, 28)):
    """
    Visualize the sizes of different layer groups (convolution, embedding, classification)
    with a horizontal stacked bar chart
    
    Parameters:
    -----------
    model : EmbedMNISTNet
        The network model to visualize
    input_size : tuple
        Input tensor shape (batch_size, channels, height, width)
    """
    def hook_fn(module, input, output):
        activation_shapes[module] = output.shape
    
    activation_shapes = {}
    hooks = []
    
    # Register hooks for all layers
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.Flatten)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass with dummy input
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    
    model.eval()
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Get sizes for each group
    conv_size = 0
    for layer in model.features:
        if layer in activation_shapes:
            if isinstance(layer, nn.Flatten):
                conv_size = activation_shapes[layer][1]  # Size after flattening
    
    embed_size = 0
    for layer in model.embedding:
        if isinstance(layer, nn.Linear):
            embed_size = layer.out_features
    
    classifier_size = 0
    for layer in model.classifier[:-1]:  # Exclude final layer
        if isinstance(layer, nn.Linear):
            classifier_size = layer.out_features
    
    # Create visualization
    plt.figure(figsize=(12, 3))
    
    # Create horizontal stacked bar
    total_size = conv_size + embed_size + classifier_size
    left = 0
    
    # Convolution part (blue)
    plt.barh(0, conv_size/total_size, left=left, color='royalblue', alpha=0.7)
    plt.text(left + (conv_size/total_size)/2, 0, f'Conv\n({conv_size})', 
             ha='center', va='center')
    left += conv_size/total_size
    
    # Embedding part (red)
    plt.barh(0, embed_size/total_size, left=left, color='crimson', alpha=0.7)
    plt.text(left + (embed_size/total_size)/2, 0, f'Embed\n({embed_size})', 
             ha='center', va='center')
    left += embed_size/total_size
    
    # Classification part (green)
    plt.barh(0, classifier_size/total_size, left=left, color='forestgreen', alpha=0.7)
    plt.text(left + (classifier_size/total_size)/2, 0, f'Classifier\n({classifier_size})', 
             ha='center', va='center')
    
    # Customize plot
    plt.yticks([])
    plt.xlabel('Proportion of Total Size')
    plt.title('Network Layer Group Sizes')
    
    # Add total size information
    plt.text(0.5, -0.5, f'Total Size: {total_size}', ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print size information
    print("\nLayer Group Sizes:")
    print(f"Convolution features: {conv_size}")
    print(f"Embedding: {embed_size}")
    print(f"Classifier hidden: {classifier_size}")
    print(f"Total size: {total_size}")

def compare_reduction_distances(results_full, results_pca, results_vae, results_importance=None, normalize=True):
    """
    Compare activation-based distances across different dimension reduction methods
    
    Parameters:
    -----------
    results_full : dict
        Results from analyze_multiple_samples for unreduced network
    results_pca : dict
        Results from analyze_multiple_samples with PCA reduction
    results_vae : dict
        Results from analyze_multiple_samples with VAE reduction
    results_importance : dict, optional
        Results from analyze_multiple_samples with importance-based reduction
    normalize : bool, default=True
        Whether to normalize distances to [0,1] range across all distances
    """
    # Get all distances for activation-based approach only
    distances = {
        'Input Distance': {},
        'Activation Distance': {},
        'Output Distance': {},
        'Combined Distance': {}
    }
    
    # Add results for each reduction method
    methods = [
        ('Unreduced', results_full),
        ('PCA', results_pca),
        ('VAE', results_vae)
    ]
    
    # Add importance results if provided
    if results_importance is not None:
        methods.append(('Importance', results_importance))
    
    # Collect distances (only from activation-based approach)
    for dist_type in distances:
        for method_name, results in methods:
            if dist_type == 'Input Distance':
                distances[dist_type][method_name] = np.mean(results['activation_based']['input_dist'])
            elif dist_type == 'Activation Distance':
                distances[dist_type][method_name] = np.mean(results['activation_based']['activation_dist'])
            elif dist_type == 'Output Distance':
                distances[dist_type][method_name] = np.mean(results['activation_based']['output_dist'])
            else:  # Combined Distance
                distances[dist_type][method_name] = np.mean(results['activation_based']['combined_dist'])
    
    # Normalize if requested
    if normalize:
        all_values = [v for d in distances.values() for v in d.values()]
        min_val = min(all_values)
        max_val = max(all_values)
        for dist_type in distances:
            for method in distances[dist_type]:
                distances[dist_type][method] = (distances[dist_type][method] - min_val) / (max_val - min_val)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))  # Number of reduction methods
    width = 0.2  # Width of bars
    multiplier = 0
    
    # Plot each distance type
    colors = ['royalblue', 'crimson', 'forestgreen', 'purple']
    for dist_type, dist_values in distances.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, list(dist_values.values()), width, 
                      label=dist_type, color=colors[multiplier])
        
        # Add value labels on bars
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', rotation=0)
        
        multiplier += 1
    
    # Customize plot
    ax.set_ylabel('Normalized Distance' if normalize else 'Mean Distance')
    ax.set_title('Activation-Based Distances Across Dimension Reduction Methods')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([method[0] for method in methods])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print numeric values
    print("\nDetailed distances:")
    for dist_type in distances:
        print(f"\n{dist_type}:")
        for method, value in distances[dist_type].items():
            print(f"{method}: {value:.3f}")

def compare_activation_distances_dim(results_conv=None, results_embedding=None, results_classification=None, results_full=None, normalize=True):
    """
    Compare activation-based distances across network sections and optionally full network
    
    Parameters:
    -----------
    results_conv : dict, optional
        Results from analyze_multiple_samples for convolution layers
    results_embedding : dict, optional
        Results from analyze_multiple_samples for embedding layer
    results_classification : dict, optional
        Results from analyze_multiple_samples for classification layers
    results_full : dict, optional
        Results from analyze_multiple_samples for full network
    normalize : bool, default=True
        Whether to normalize distances to [0,1] range across all distances
    """
    # Get all distances for activation-based approach
    distances = {
        'Input Distance': {},
        'Activation Distance': {},
        'Output Distance': {},
        'Combined Distance': {}
    }
    
    # Add section results if provided
    sections = []
    if results_conv:
        sections.append(('Convolution', results_conv))
    if results_embedding:
        sections.append(('Embedding', results_embedding))
    if results_classification:
        sections.append(('Classification', results_classification))
    if results_full:
        sections.append(('Full Network', results_full))
    
    # Collect distances
    for dist_type in distances:
        for section_name, results in sections:
            if dist_type == 'Input Distance':
                distances[dist_type][section_name] = np.mean(results['activation_based']['input_dist'])
            elif dist_type == 'Activation Distance':
                distances[dist_type][section_name] = np.mean(results['activation_based']['activation_dist'])
            elif dist_type == 'Output Distance':
                distances[dist_type][section_name] = np.mean(results['activation_based']['output_dist'])
            else:  # Combined Distance
                distances[dist_type][section_name] = np.mean(results['activation_based']['combined_dist'])
    
    # Normalize if requested
    if normalize:
        all_values = [v for d in distances.values() for v in d.values()]
        min_val = min(all_values)
        max_val = max(all_values)
        for dist_type in distances:
            for section in distances[dist_type]:
                distances[dist_type][section] = (distances[dist_type][section] - min_val) / (max_val - min_val)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sections))  # Number of sections
    width = 0.2  # Width of bars
    multiplier = 0
    
    # Plot each distance type
    for dist_type, dist_values in distances.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, list(dist_values.values()), width, label=dist_type)
        
        # Add value labels on bars
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', rotation=0)
        
        multiplier += 1
    
    # Customize plot
    ax.set_ylabel('Normalized Distance' if normalize else 'Mean Distance')
    ax.set_title('Activation-Based Distances Across Network Sections')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([section[0] for section in sections])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print numeric values
    print("\nDetailed distances (activation-based approach):")
    for dist_type in distances:
        print(f"\n{dist_type}:")
        for section, value in distances[dist_type].items():
            print(f"{section}: {value:.3f}")


def find_important_neurons_in_range(activations, start_idx, end_idx, n_neurons=None):
    """
    Find neuron importance scores based on variance and activity
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Array of shape [n_samples, n_features] containing activation values
    start_idx : int
        Start index of the range to analyze
    end_idx : int
        End index of the range to analyze
    n_neurons : int, optional
        Number of neurons to return. If None, returns scores for all neurons
        
    Returns:
    --------
    numpy.ndarray
        Indices of neurons sorted by importance (most to least important)
    numpy.ndarray
        Importance scores for all neurons
    """
    # Extract the range we want to analyze
    activations_subset = activations[:, start_idx:end_idx]
    
    # Calculate variance (activity spread)
    variance_scores = np.var(activations_subset, axis=0)
    
    # Calculate mean absolute activation (activity magnitude)
    magnitude_scores = np.mean(np.abs(activations_subset), axis=0)
    
    # Normalize scores
    variance_norm = (variance_scores - variance_scores.min()) / (variance_scores.max() - variance_scores.min() + 1e-10)
    magnitude_norm = (magnitude_scores - magnitude_scores.min()) / (magnitude_scores.max() - magnitude_scores.min() + 1e-10)
    
    # Combine scores
    importance_scores = variance_norm + magnitude_norm
    
    # Sort neurons by importance (highest to lowest)
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    if n_neurons is not None:
        sorted_indices = sorted_indices[:n_neurons]
    
    # Adjust indices to match original activation array
    original_indices = sorted_indices + start_idx
    
    return original_indices, importance_scores

def calculate_accuracy(model, loader, device='cuda'):
    """
    Calculate overall accuracy of the model on given data loader
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to evaluate
    loader : DataLoader
        DataLoader containing validation or test data
    device : str, default='cuda'
        Device to run evaluation on
        
    Returns:
    --------
    float
        Accuracy as percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def create_pruned_model(original_model, indices_to_zero, device='cuda'):
    """
    Create a copy of the model and zero out weights for specified neurons
    
    Parameters:
    -----------
    original_model : torch.nn.Module
        The original neural network model
    indices_to_zero : numpy.ndarray or list
        Indices of neurons to zero out
    device : str, default='cuda'
        Device to place the model on
        
    Returns:
    --------
    torch.nn.Module
        New model with zeroed weights at specified indices
    """
    # Create a deep copy of the model
    pruned_model = copy.deepcopy(original_model)
    pruned_model = pruned_model.to(device)
    
    # Set model to eval mode
    pruned_model.eval()
    
    # Get the layers of the model
    layers = list(pruned_model.model)
    
    # Find all Linear layers
    linear_layers = [(i, layer) for i, layer in enumerate(layers) 
                    if isinstance(layer, nn.Linear)]
    
    # For each specified index, zero out the corresponding weights and bias
    for idx in indices_to_zero:
        # Calculate which layer this neuron belongs to
        current_pos = 0
        for layer_idx, layer in linear_layers:
            output_size = layer.out_features
            
            # Check if the index falls within this layer's outputs
            if current_pos <= idx < current_pos + output_size:
                # Calculate the relative position within this layer
                relative_idx = idx - current_pos
                
                # Zero out weights and bias for this neuron
                layer.weight.data[relative_idx].zero_()
                if layer.bias is not None:
                    layer.bias.data[relative_idx] = 0
                break
                
            current_pos += output_size
    
    return pruned_model

def split_activations_by_class(activations, labels):
    """
    Split activation data by class label
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Array of activations with shape [n_samples, n_features]
    labels : numpy.ndarray
        Array of class labels with shape [n_samples]
        
    Returns:
    --------
    dict
        Dictionary containing activations for each class
        {class_label: class_activations_array}
    """
    # Initialize dictionary to store class-specific activations
    class_activations = {}
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Split activations by class
    for class_label in unique_classes:
        # Create mask for current class
        class_mask = labels == class_label
        # Store activations for current class
        class_activations[class_label] = activations[class_mask]
    
    return class_activations

def evaluate_per_class(model, test_loader, device='cuda'):
    """
    Evaluate model accuracy for each MNIST digit class
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model to evaluate
    test_loader : DataLoader
        DataLoader containing test data
    device : str, default='cuda'
        Device to run evaluation on
        
    Returns:
    --------
    dict
        Dictionary containing per-class accuracies
    """
    model.eval()
    # Initialize counters for each class
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update counters for each class
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
    
    # Calculate and print accuracy for each class
    print("\nAccuracy per class:")
    for i in range(10):
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f"Digit {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return {i: 100 * class_correct[i] / class_total[i] for i in range(10)}

def evaluate_pruning_strategies(model, test_loader, activations, num_steps=20, max_neurons=256, device='cuda', seed=42):
    """
    Compare pruning based on importance vs random selection
    """
    np.random.seed(seed)
    total_neurons = activations.shape[1]
    neurons_per_step = max_neurons // num_steps
    
    results = {
        'importance_based': {'neurons': [], 'accuracy': []},
        'random': {'neurons': [], 'accuracy': []}
    }
    
    # Get importance-based ordering of all neurons
    indices, scores = find_important_neurons_in_range(
        activations=activations,
        start_idx=0,
        end_idx=total_neurons
    )
    
    # Indices are already sorted from most to least important
    importance_order = indices
    
    # Create random ordering
    random_order = np.random.permutation(total_neurons)
    
    # Store baseline accuracy
    baseline_acc = calculate_accuracy(model, test_loader, device)
    results['importance_based']['neurons'].append(0)
    results['importance_based']['accuracy'].append(baseline_acc)
    results['random']['neurons'].append(0)
    results['random']['accuracy'].append(baseline_acc)
    
    # Evaluate progressively larger sets of pruned neurons
    for i in range(num_steps):
        n_neurons = (i + 1) * neurons_per_step
        
        # Importance-based pruning (prune least important neurons)
        neurons_to_prune = importance_order[-n_neurons:]
        pruned_model = create_pruned_model(model, neurons_to_prune, device)
        importance_acc = calculate_accuracy(pruned_model, test_loader, device)
        
        # Random pruning
        random_neurons = random_order[:n_neurons]
        random_pruned_model = create_pruned_model(model, random_neurons, device)
        random_acc = calculate_accuracy(random_pruned_model, test_loader, device)
        
        # Store results
        results['importance_based']['neurons'].append(n_neurons)
        results['importance_based']['accuracy'].append(importance_acc)
        results['random']['neurons'].append(n_neurons)
        results['random']['accuracy'].append(random_acc)
        
        print(f"Pruned {n_neurons} neurons:")
        print(f"  Importance-based accuracy: {importance_acc:.2f}%")
        print(f"  Random pruning accuracy: {random_acc:.2f}%")
    
    return results

def plot_pruning_comparison(results):
    """
    Plot comparison of pruning strategies
    
    Parameters:
    -----------
    results : dict
        Results from evaluate_pruning_strategies
    """
    plt.figure(figsize=(10, 6))
    
    # Plot both strategies
    plt.plot(results['importance_based']['neurons'], 
             results['importance_based']['accuracy'], 
             'b-', label='Importance-based pruning')
    plt.plot(results['random']['neurons'], 
             results['random']['accuracy'], 
             'r--', label='Random pruning')
    
    plt.xlabel('Number of Pruned Neurons')
    plt.ylabel('Model Accuracy (%)')
    plt.title('Comparison of Pruning Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines at specific accuracy drops
    baseline = results['importance_based']['accuracy'][0]
    thresholds = [0.99, 0.95, 0.90]
    colors = ['g', 'y', 'r']
    
    for threshold, color in zip(thresholds, colors):
        threshold_acc = baseline * threshold
        
        # Find crossing points for both strategies
        for strategy in ['importance_based', 'random']:
            accuracies = np.array(results[strategy]['accuracy'])
            neurons = np.array(results[strategy]['neurons'])
            
            # Find where accuracy drops below threshold
            if any(accuracies < threshold_acc):
                idx = np.where(accuracies < threshold_acc)[0][0]
                plt.axvline(x=neurons[idx], color=color, alpha=0.3, linestyle=':')
                plt.text(neurons[idx], plt.ylim()[0], 
                        f'{int(threshold*100)}%\n{neurons[idx]}n',
                        rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

def compare_pruning_strategies(model, test_loader, important_indices, device='cuda'):
    """
    Compare pruning using important neurons vs random neurons
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    test_loader : DataLoader
        Test data loader
    important_indices : list or numpy.ndarray
        List of neuron indices sorted by importance (most to least important)
    device : str, default='cuda'
        Device to run evaluation on
    """
    n_neurons = len(important_indices)
    
    # Create random indices
    random_indices = np.random.randint(0, 512, size=len(important_indices))

    
    # Initialize results
    results = {
        'n_pruned': [0],  # Start with 0 pruned neurons
        'importance_acc': [calculate_accuracy(model, test_loader, device)],  # Baseline accuracy
        'random_acc': [calculate_accuracy(model, test_loader, device)]  # Baseline accuracy
    }
    
    # Evaluate progressively larger sets of pruned neurons
    for i in range(n_neurons):
        # Importance-based pruning (prune least important first)
        importance_neurons = important_indices[-(i+1):]  # Take last n neurons
        pruned_model = create_pruned_model(model, importance_neurons, device)
        importance_acc = calculate_accuracy(pruned_model, test_loader, device)
        
        # Random pruning
        random_neurons = random_indices[:i+1]  # Take first n neurons
        random_model = create_pruned_model(model, random_neurons, device)
        random_acc = calculate_accuracy(random_model, test_loader, device)
        
        # Store results
        results['n_pruned'].append(i+1)
        results['importance_acc'].append(importance_acc)
        results['random_acc'].append(random_acc)
        
        print(f"Pruned {i+1}/{n_neurons} neurons:")
        print(f"  Importance-based accuracy: {importance_acc:.2f}%")
        print(f"  Random pruning accuracy: {random_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['n_pruned'], results['importance_acc'], 'b-', label='Importance-based pruning')
    plt.plot(results['n_pruned'], results['random_acc'], 'r--', label='Random pruning')
    plt.xlabel('Number of Pruned Neurons')
    plt.ylabel('Model Accuracy (%)')
    plt.title('Comparison of Pruning Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results