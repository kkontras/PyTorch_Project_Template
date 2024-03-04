import torch
import logging
import copy
from torchmetrics import F1Score, CohenKappa, Accuracy
from collections import defaultdict
import torchmetrics
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
class All_Evaluator:
    def __init__(self, config, dataloaders: dict):


        if config.get("task", "classification") == "classification":
            evaluator_class = globals()["General_Evaluator"]
        elif config.get("task", "classification") == "regression":
            evaluator_class = globals()["General_Evaluator_Regression"]

        # self.train_evaluator = General_Evaluator_Regression(config, len(dataloaders.train_loader.dataset))
        # self.val_evaluator = General_Evaluator_Regression(config, len(dataloaders.train_loader.dataset))
        # if hasattr(dataloaders, "test_loader"):
        #     self.test_evaluator = General_Evaluator_Regression(config, len(dataloaders.test_loader.dataset))

        self.train_evaluator = evaluator_class(config, len(dataloaders.train_loader.dataset))
        self.val_evaluator = evaluator_class(config, len(dataloaders.train_loader.dataset))
        if hasattr(dataloaders, "test_loader"):
            self.test_evaluator = evaluator_class(config, len(dataloaders.test_loader.dataset))

class General_Evaluator:
    def __init__(self, config, total_instances: int):
        self.config = config
        self.total_instances = total_instances
        self.num_classes = config.model.args.num_classes
        self.reset()

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        logging.info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def set_early_stop(self):
        self.early_stop = True

    def reset(self):
        self.losses = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w if self.config.model.args.multi_loss.multi_supervised_w[pred_key] != 0.0}
        self.labels = []
        self.processed_instances = 0

    def process(self, all_output: dict):

        logits = {pred: all_output["pred"][pred].cpu() for pred in all_output["pred"]}
        label = all_output["label"].cpu()
        loss = {l_i: all_output["loss"][l_i].detach().cpu() for l_i in all_output["loss"]}
        num_instances = label.shape[0]

        for pred_key in logits:
            if pred_key not in self.preds:
                continue
            assert (len(logits[pred_key].shape) == 2), "The shape of logits must be in format [bs, num_test_clips * num_test_crops, total_classes]"
            self.preds[pred_key].append(logits[pred_key])

        self.labels.append(label)

        self.processed_instances += num_instances
        self.losses.append(loss)

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def mean_batch_loss(self):
        if len(self.losses)==0:
            return None, ""
        mean_batch_loss = {}
        for key in self.losses[0].keys():
            mean_batch_loss[key] = torch.stack([self.losses[i][key] for i in range(len(self.losses))]).mean().item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self):


        targets_tens = torch.concatenate(self.labels).cpu().flatten()

        mean_batch_loss, _ = self.mean_batch_loss()

        total_preds, metrics  = {}, defaultdict(dict)
        if mean_batch_loss is not None:
            metrics["loss"] = mean_batch_loss

        ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="MULTICLASS")
        for pred_key in self.preds:
            if len(self.preds[pred_key]) == 0:
                continue
            total_preds = torch.concatenate(self.preds[pred_key]).cpu()#[:self.processed_instances]
            metrics["acc"][pred_key] = Accuracy(task="multiclass", num_classes=self.num_classes)(total_preds,targets_tens).item()
            if self.num_classes > 5:
                metrics["top5_acc"][pred_key] = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)(total_preds,
                                                                                                     targets_tens).item()
            metrics["f1"][pred_key] = F1Score( task="multiclass", num_classes=self.num_classes, average='macro')(total_preds, targets_tens).item()
            metrics["f1_mi"][pred_key] = F1Score( task="multiclass", num_classes=self.num_classes, average='micro')(total_preds, targets_tens).item()
            metrics["k"][pred_key] = CohenKappa(task="multiclass", num_classes=self.num_classes)(total_preds, targets_tens).item()
            metrics["f1_perclass"][pred_key] = F1Score(task="multiclass", num_classes=self.num_classes, average=None)(total_preds, targets_tens)
            metrics["ece"][pred_key] = ece(total_preds, targets_tens).item()

        metrics = dict(metrics) #Avoid passing empty dicts to logs, better return an error!

        return metrics

    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        # Flag if its saved dont save it again on $save_every
        not_saved = True
        validate_with = self.config.early_stopping.get("validate_with", "loss")
        if validate_with == "loss":
            is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])
        elif validate_with == "accuracy":
            is_best = (metrics["acc"]["combined"] > best_logs["acc"]["combined"])
        else:
            raise ValueError("self.agent.config.early_stopping.validate_with should be either loss or accuracy")
        return is_best

class General_Evaluator_Regression:
    def __init__(self, config, total_instances: int):
        self.config = config
        self.total_instances = total_instances
        self.num_classes = config.model.args.num_classes
        self.reset()

        self.early_stop = False

        self.best_acc = 0.0
        self.best_loss = 0.0

    def set_best(self, best_acc, best_loss):
        self.best_acc = best_acc
        self.best_loss = best_loss
        logging.info("Set current best acc {}, loss {}".format(self.best_acc, self.best_loss))

    def reset(self):
        self.losses = []
        self.preds = {pred_key.lower(): [] for pred_key in self.config.model.args.multi_loss.multi_supervised_w if self.config.model.args.multi_loss.multi_supervised_w[pred_key] != 0.0}
        self.labels = []
        self.processed_instances = 0

    def set_early_stop(self):
        self.early_stop = True

    def process(self, all_output: dict):

        logits = all_output["pred"]
        label = all_output["label"]
        loss = all_output["loss"]
        num_instances = label.shape[0]

        for pred_key in logits:
            if pred_key not in self.preds:
                continue
            assert (len(logits[pred_key].shape) == 2), "The shape of logits must be in format [bs, num_test_clips * num_test_crops, total_classes]"
            # print("logits[pred_key].shape", logits[pred_key].shape)
            # print(logits[pred_key].shape)
            # print(self.preds[pred_key][self.processed_instances : self.processed_instances + num_instances].shape)
            # self.preds[pred_key][self.processed_instances : self.processed_instances + num_instances] = logits[pred_key]
            self.preds[pred_key].append(logits[pred_key])

        # self.labels[self.processed_instances : self.processed_instances + num_instances] = label
        self.labels.append(label)

        self.processed_instances += num_instances
        self.losses.append(loss)

    def get_early_stop(self):
        return self.early_stop

    def enable_early_stop(self):
        self.early_stop = True

    def mean_batch_loss(self):
        if len(self.losses)==0:
            return None, ""
        mean_batch_loss = {}
        for key in self.losses[0].keys():
            mean_batch_loss[key] = torch.stack([self.losses[i][key] for i in range(len(self.losses))]).mean().item()

        message = ""
        for mean_key in mean_batch_loss: message += "{}: {:.3f} ".format(mean_key, mean_batch_loss[mean_key])

        return dict(mean_batch_loss), message

    def evaluate(self):


        targets_tens = torch.concatenate(self.labels).cpu().flatten()

        mean_batch_loss, _ = self.mean_batch_loss()

        total_preds, metrics  = {}, defaultdict(dict)
        if mean_batch_loss is not None:
            metrics["loss"] = mean_batch_loss

        ece = torchmetrics.CalibrationError(num_classes=self.config.model.args.num_classes, task="BINARY")
        for pred_key in self.preds:
            total_preds = torch.concatenate(self.preds[pred_key]).cpu().squeeze()#[:self.processed_instances]

            binary_truth = (targets_tens[targets_tens!=0] > 0)
            binary_preds = (total_preds[targets_tens!=0] > 0)
            metrics["acc"][pred_key] = Accuracy(task="binary")(binary_preds,binary_truth).item()

            # binary_truth = (targets_tens > 0)
            # binary_preds = (total_preds > 0)
            # metrics["acc"][pred_key] = Accuracy(task="binary")(binary_preds,binary_truth).item()

            metrics["f1"][pred_key] = f1_score(binary_preds.cpu().numpy(), binary_truth.cpu().numpy(), average='weighted')

            test_preds = total_preds.view(-1).cpu().detach().numpy()
            test_truth = targets_tens.view(-1).cpu().detach().numpy()


            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            metrics["mae"][pred_key] = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
            metrics["corr"][pred_key] = np.corrcoef(test_preds, test_truth)[0][1]
            metrics["acc_7"][pred_key] = multiclass_acc(test_preds_a7, test_truth_a7)
            metrics["acc_5"][pred_key] = multiclass_acc(test_preds_a5, test_truth_a5)


            # metrics["acc"][pred_key] = Accuracy(task="multiclass", num_classes=self.num_classes)(total_preds,targets_tens).item()
            # if self.num_classes > 5:
            #     metrics["top5_acc"][pred_key] = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=5)(total_preds,
            #                                                                                          targets_tens).item()
            # metrics["f1"][pred_key] = F1Score( task="multiclass", num_classes=self.num_classes, average='macro')(total_preds, targets_tens).item()
            # metrics["f1_mi"][pred_key] = F1Score( task="multiclass", num_classes=self.num_classes, average='micro')(total_preds, targets_tens).item()
            # metrics["k"][pred_key] = CohenKappa(task="multiclass", num_classes=self.num_classes)(total_preds, targets_tens).item()
            # metrics["f1_perclass"][pred_key] = F1Score(task="multiclass", num_classes=self.num_classes, average=None)(total_preds, targets_tens)
            # metrics["ece"][pred_key] = ece(total_preds, targets_tens).item()

        metrics = dict(metrics) #Avoid passing empty dicts to logs, better return an error!

        return metrics

    def is_best(self, metrics = None, best_logs=None):
        if metrics is None:
            metrics = self.evaluate()

        # Flag if its saved dont save it again on $save_every
        not_saved = True
        validate_with = self.config.early_stopping.get("validate_with", "loss")
        if validate_with == "loss":
            is_best = (metrics["loss"]["total"] < best_logs["loss"]["total"])
        elif validate_with == "accuracy":
            is_best = (metrics["acc"]["combined"] > best_logs["acc"]["combined"])
        else:
            raise ValueError("self.agent.config.early_stopping.validate_with should be either loss or accuracy")
        return is_best
