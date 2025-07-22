# Code adapted from https://github.com/LAION-AI/CLIP_benchmark/tree/main/clip_benchmark
#
# Code adapted from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
# Thanks to the authors of OpenCLIP

from contextlib import suppress
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import RetrievalMAP
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score
from collections import defaultdict


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                texts = templates[classname]
            elif type(templates) == list:
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)
            try:
                class_embeddings = model.encode_text(texts)
            except AttributeError:
                class_embeddings = model.inference_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True, one_class=True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    class_correct = defaultdict(int)  # ADDED 0 (default value provided by int())
    class_total = defaultdict(int)
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                try:
                    image_features = model.encode_image(images)
                    if isinstance(image_features, tuple):
                        image_features = image_features[0]
                except AttributeError:
                    image_features = model.inference_vision(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            true.append(target.cpu())
            pred.append(logits.float().cpu())

            if one_class:
                _, preds = torch.max(logits, 1)  # ADDED
                for label, predx in zip(target, preds):  # ADDED
                    if label == predx:  # ADDED
                        class_correct[label.item()] += 1  # ADDED
                    class_total[label.item()] += 1  # ADDED

    pred = torch.cat(pred)
    true = torch.cat(true)
    if one_class:
        class_accuracies = {classname: class_correct[i] / class_total[i] for i, classname in
                            enumerate(dataloader.dataset.classes)}  # ADDED
        return pred, true, class_accuracies
    else:
        return pred, true


def map_per_class(scores, targets, topk=100):
    map_per_class = {}
    for k in range(scores.size(1)):
        scores_k = scores[:, k]
        cls_indexes = torch.zeros(scores.size(0), dtype=torch.long)
        cls_relevance = (targets == k)
        rmap = RetrievalMAP(top_k=topk)
        cls_map = rmap(preds=scores_k, target=cls_relevance, indexes=cls_indexes)
        map_per_class[k] = cls_map.item()
    return map_per_class


def map_per_class_ml(scores, targets, topk=100):
    map_per_class = {}
    for k in range(scores.size(1)):
        scores_k = scores[:, k]
        cls_relevance = targets[:, k]
        cls_indexes = torch.zeros(scores.size(0), dtype=torch.long)
        rmap = RetrievalMAP(top_k=topk)
        cls_map = rmap(preds=scores_k, target=cls_relevance, indexes=cls_indexes)
        map_per_class[k] = cls_map.item()
    return map_per_class


def multilabel_accuracy(true_labels, predicted_scores, other_features=False):
    """
    Calculate the  accuracy for multi-label classification where k is the number of true labels for each sample.
    
    Parameters:
        true_labels (torch.Tensor): Binary matrix of shape (n_samples, n_classes)
        predicted_scores (torch.Tensor): Matrix of predicted scores of shape (n_samples, n_classes)
    
    Returns:
        float: The top-k accuracy of the model
    """
    if other_features:  # only for multilabel classification
        last_values = predicted_scores[:, -1].unsqueeze(1)

        # Compare each value in the row with the last value
        result = (predicted_scores > last_values).int()

        # Ensure that the last column in the result is set to 0
        result[:, -1] = 0

        class_accuracies = {}

        # Iterate over each of the 20 classes (columns)
        for class_idx in range(result.shape[1]):
            # Compare the values in the result and target tensors for the current class
            correct_predictions = (result[:, class_idx] == true_labels[:, class_idx]).sum().item()

            # Calculate the accuracy as the ratio of correct predictions to the total number of samples (800)
            accuracy = correct_predictions / result.shape[0]

            # Store the accuracy in the dictionary
            class_accuracies[f'{class_idx}'] = accuracy

        return class_accuracies
    else:

        num_classes = predicted_scores.shape[1]
        predicted_labels = torch.zeros_like(predicted_scores, dtype=torch.int)  # Placeholder for binary predictions

        for class_idx in range(num_classes):
            # Compute the average score of all other classes, excluding the current class
            other_classes_mask = torch.ones_like(predicted_scores, dtype=torch.bool)
            other_classes_mask[:, class_idx] = False  # Exclude current class
            average_other_classes = predicted_scores[other_classes_mask].reshape(predicted_scores.shape[0], -1).mean(
                dim=1, keepdim=True)

            # Make binary predictions: 1 if class score > average of other class scores
            predicted_labels[:, class_idx] = (predicted_scores[:, class_idx] > average_other_classes.squeeze()).int()

        class_accuracies = {}
        # Compute accuracy for each class
        for class_idx in range(num_classes):
            correct_predictions = (predicted_labels[:, class_idx] == true_labels[:, class_idx]).sum().item()
            accuracy = correct_predictions / predicted_labels.shape[0]
            class_accuracies[f'Class {class_idx} Accuracy'] = accuracy

        return class_accuracies


def classwise_f1_scores(predicted_scores, true_labels, classes, other_features=False):
    """
    Calculate per-class F1-score and count the number of predictions per class.

    Uses the same approach as the provided accuracy function, where predictions 
    are determined based on whether they are greater than the last value in the row.

    Parameters:
        true_labels (torch.Tensor): Binary matrix of shape (n_samples, n_classes)
        predicted_scores (torch.Tensor): Matrix of predicted scores of shape (n_samples, n_classes)

    Returns:
        dict: Class-wise F1 scores
        dict: Number of predictions per class (TP + FP)
    """

    if other_features:  # only for multilabel classification
        last_values = predicted_scores[:, -1].unsqueeze(1)  # Get the last column's values
        predicted_labels = (predicted_scores > last_values).int()  # Convert scores to binary
        predicted_labels[:, -1] = 0  # Ensure last column is always 0
        f1_report = classification_report(true_labels[:, :-1], predicted_labels[:, :-1], digits=4,
                                          target_names=classes[:-1])
    else:
        num_classes = predicted_scores.shape[1]
        predicted_labels = torch.zeros_like(predicted_scores, dtype=torch.int)  # Placeholder for binary predictions

        for class_idx in range(num_classes):
            # Compute the average score of all other classes, excluding the current class
            other_classes_mask = torch.ones_like(predicted_scores, dtype=torch.bool)
            other_classes_mask[:, class_idx] = False  # Exclude current class
            average_other_classes = predicted_scores[other_classes_mask].reshape(predicted_scores.shape[0], -1).mean(
                dim=1, keepdim=True)

            # Make binary predictions: 1 if class score > average of other class scores
            predicted_labels[:, class_idx] = (predicted_scores[:, class_idx] > average_other_classes.squeeze()).int()
        print(true_labels.shape, predicted_labels.shape)
        f1_report = classification_report(true_labels, predicted_labels, digits=4, target_names=classes)

    return f1_report


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=True, save_clf=None,
             load_clfs=[], one_class=True, other_features=False):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)

    if save_clf is not None:
        torch.save(classifier, save_clf)

    if one_class:
        logits, target, class_accuracies = run_classification(model, classifier, dataloader, device, amp=amp)
    else:
        logits, target = run_classification(model, classifier, dataloader, device, amp=amp, one_class=False)
    is_multilabel = (len(target.shape) == 2)
    cls_map = {}
    cls_acc = {}
    cls_rmap = {}
    if is_multilabel:
        print("Detected a multi-label classification dataset")
        per_class_acc = multilabel_accuracy(target, logits, other_features=other_features)
        # Multiple labels per image, multiple classes on the dataset
        mAP = map_per_class_ml(logits, target, topk=100)  # multilabel
        for class_name, acc in zip(dataloader.dataset.classes, per_class_acc.values()):
            print(f"Class: {class_name}, Accuracy: {acc}")
            cls_acc[f"{class_name}_acc"] = acc
        if other_features:
            cls_acc["Average_class_accuracy"] = np.mean(list(per_class_acc.values())[:-1])
        else:
            cls_acc["Average_class_accuracy"] = np.mean(list(per_class_acc.values()))
        for class_name, rmap in zip(dataloader.dataset.classes, mAP.values()):
            print(f"Class: {class_name}, mAP: {rmap}")
            cls_rmap[f"{class_name}_mAP"] = rmap
        if other_features:
            cls_rmap["Ave_mAP"] = np.mean(list(mAP.values())[:-1])
        else:
            cls_rmap["Ave_mAP"] = np.mean(list(mAP.values()))
        f1_report = classwise_f1_scores(logits, target, dataloader.dataset.classes, other_features=other_features)
        print("F1 report", f1_report)
        return {**{key: value for key, value in cls_acc.items()}, **{key: value for key, value in cls_rmap.items()},
                "F1 report: ": f1_report}
    else:
        map_cls = map_per_class(logits, target, topk=100)
        for class_name, ap in zip(dataloader.dataset.classes, map_cls.values()):
            print(f"Class: {class_name}, mAP: {ap}")
            cls_map[f"{class_name}_maP"] = ap
        cls_map["mAP"] = np.mean(list(map_cls.values()))

        pred = logits.argmax(axis=1)

        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan")
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            f1_report = classification_report(target, pred, digits=4, target_names=dataloader.dataset.classes)
            print("F1 report: ", f1_report)
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall,
                **{key: value for key, value in class_accuracies.items()},
                **{key: value for key, value in cls_map.items()}, "classification report: ": f1_report}
