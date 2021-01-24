import torch
import torch.nn.functional as F

def compute_uncertainties(net, X, K=100):
    """
    @param net: Pytorch model representing the bayesian network
    @param X: Pytorch tensor of shape (n_samples, sample_shape...)
    @param K: int representing the number of predictions to be done for each single sample
    @return predictions_uncertainty: pytorch tensor of shape (n_samples,) containing for each sample
                                        the predicted class (that is the class with highest mean value
                                        along K predictions)
    @return predicted_class_variance: pytorch tensor of shape (n_samples,) containing for each sample the
                                        variance associated to the predicted class of that sample
    """
    p_hat = []
    for k in range(K):
        scores = net(X)
        scores = scores if isinstance(net, torch.nn.Module) else scores[0]
        # scores.shape == (n_samples, n_classes)
        scores = F.softmax(scores, dim=1)
        p_hat.append(scores)

    p_hat = torch.stack(p_hat)
    # p_hat.shape == (K, n_samples, n_classes)

    # Mean over MC(MonteCarlo) samples
    ## Per ogni campione per ogni classe calcolo la media lungo le K iterazioni
    # For each sample for each class compute the mean along the K predictions
    mean_probs_over_draw = torch.mean(p_hat, dim=0)
    # mean_probs_over_draw.shape == (n_samples, n_classes)

    # For each sample, the predicted class is the one having the highest mean value
    predictions_uncertainty = torch.argmax(mean_probs_over_draw, dim=1)
    # predictions_uncertainty.shape == (n_samples,)

    aleatoric = torch.mean(p_hat * (1 - p_hat), dim=0)
    epistemic = torch.mean(p_hat ** 2, dim=0) - torch.mean(p_hat, dim=0) ** 2

    uncertainties_among_labels = epistemic + aleatoric
    # uncertainties_among_labels.shape == (n_samples, n_classes)

    predicted_class_variances = torch.tensor([uncertainty[prediction] for prediction, uncertainty in
                                              zip(predictions_uncertainty, uncertainties_among_labels)])
    # predicted_class_variances.shape == (n_samples, )

    return predictions_uncertainty, predicted_class_variances

def compute_uncertainties_softmax(net, X):
    """
    Softmax std per single(i.e. K=1) predictions --> Softmax uncertainty
    Use case: once we have a network trained on a certain dataset, at inference time we fed to the network
    samples from another dataset and check how much uncertain is the network about these samples.
    The complete uncertainty(i.e. the ideal case) should be predicting the same probability
    value along all of the classes(i.e. low std)
    @param net: Pytorch model representing the deterministic network
    @param X: Pytorch tensor of shape (n_samples, sample_shape...)
    @return std_predictions: Pytorch tensor of shape (n_samples, ) containing the std associated to the prediction
    """
    scores = net(X)      # scores.shape == (n_samples, n_classes)
    scores = scores if isinstance(net, torch.nn.Module) else scores[0]
    scores = F.softmax(scores, dim=1)
    std_predictions = torch.std(scores, axis=1)
    return std_predictions