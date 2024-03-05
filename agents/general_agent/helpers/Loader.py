import torch
import torch.nn as nn

from models.OGM_Model import *

from colorama import Fore, Back, Style
import torch.optim as optim
from utils.schedulers.no_scheduler import No_Scheduler
from utils.schedulers.warmup_scheduler import WarmupScheduler
import wandb
import os

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=False
import logging

logger = logging.getLogger('torch._dynamo.symbolic_convert:')
logger.setLevel(logging.WARNING)

logger = logging.getLogger('torch._dynamo.output_graph:')
logger.setLevel(logging.WARNING)

class Loader():

    def __init__(self, agent):
        self.agent = agent

    def load_pretrained_models(self):
        if "pretrained_model" in self.agent.config.model:
            if self.agent.config.model.pretrained_model["use"] and not self.agent.config.model.load_ongoing:
                if self.agent.accelerator.ismainprocess:
                    self.agent.logger.info("Loading pretrained model from file {}".format(self.agent.config.model.pretrained_model["dir"]))
                checkpoint = torch.load(self.agent.config.model.pretrained_model["dir"])
                self.agent.model.load_state_dict(checkpoint["model_state_dict"])

    def load_models_n_optimizer(self):

        enc = self.sleep_load_encoder(enc_args=self.agent.config.model.get("encoders", []))
        model_class = globals()[self.agent.config.model.model_class]
        # self.agent.model = model_class(enc, args = self.agent.config.model.args)
        # self.agent.model = nn.DataParallel(model_class(encs = enc, args = self.agent.config.model.args), device_ids=[torch.device(i) for i in self.agent.config.training_params.gpu_device])

        if "save_base_dir" in self.agent.config.model and "swin_backbone" in self.agent.config.model.args:
            self.agent.config.model.args.swin_backbone = os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.args.swin_backbone)

        if "save_base_dir" in self.agent.config.model and "pretraining_paths" in self.agent.config.model.args:
            self.agent.config.model.args.pretraining_paths = {i: os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.args.pretraining_paths[i]) for i in self.agent.config.model.args.pretraining_paths}


        self.agent.model = model_class(encs = enc, args = self.agent.config.model.args)
        self.agent.model.cuda()

        if self.agent.config.model.get("compile_model", False):
            # self.agent.model = torch.compile(self.agent.model, mode="reduce-overhead" )
            # self.agent.model = torch.compile(self.agent.model, mode="max-autotune", dynamic=True)
            self.agent.model = torch.compile(self.agent.model, backend="eager")

        self._my_numel(self.agent.model, verbose=True)
        if self.agent.config.optimizer.type == "Adam":

            self.agent.optimizer = optim.Adam(self.agent.model.parameters(),
                                              lr=self.agent.config.optimizer.learning_rate,
                                              betas=(self.agent.config.optimizer.beta1, self.agent.config.optimizer.beta2),
                                              eps=1e-07,
                                              weight_decay=self.agent.config.optimizer.weight_decay)
        elif self.agent.config.optimizer.type == "SGD":
            self.agent.optimizer = optim.SGD(self.agent.model.parameters(),
                                    lr=self.agent.config.optimizer.learning_rate,
                                    weight_decay=self.agent.config.optimizer.weight_decay,
                                    momentum=self.agent.config.optimizer.momentum)
        elif self.agent.config.optimizer.type == "Adadelta":
            self.agent.optimizer = optim.Adadelta(self.agent.model.parameters(),
                                            lr=self.agent.config.optimizer.learning_rate,
                                            rho=0.9,
                                            eps=1e-06,
                                            weight_decay=self.agent.config.optimizer.weight_decay)
        elif self.agent.config.optimizer.type == "Adaw":
            ind_opt = self.agent.config.optimizer.get("indepentent_params", False)
            if ind_opt is not False:
                list_of_params = []
                for i, key in enumerate(ind_opt):
                    name = "mod{}_{}_model".format(i, key)
                    if hasattr(self.agent.model, name):
                        list_of_params.append({'params':getattr(self.agent.model, name).parameters(), "lr": ind_opt[key]["learning_rate"], "weight_decay": ind_opt[key]["weight_decay"]})
                if hasattr(self.agent.model, "classifier"):
                    list_of_params.append({'params':self.agent.model.classifier.parameters(), "lr": self.agent.config.optimizer.learning_rate, "weight_decay": self.agent.config.optimizer.weight_decay})
                self.agent.optimizer = optim.AdamW(list_of_params,
                                            lr=self.agent.config.optimizer.learning_rate,
                                            weight_decay=self.agent.config.optimizer.weight_decay)
            else:
                self.agent.optimizer = optim.AdamW(self.agent.model.parameters(),
                                            lr=self.agent.config.optimizer.learning_rate,
                                            weight_decay=self.agent.config.optimizer.weight_decay)

        self.load_pretrained_models()

    def load_best_model(self):

        file_name = self.agent.config.model.save_dir
        if "data_split" in self.agent.config.dataset and self.agent.config.dataset.data_split.get("split_method",
                                                                                                  False) == "patients_folds":
            file_name = file_name.format(self.agent.config.dataset.data_split.fold)

        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, file_name)

        if os.path.exists(file_name):
            prev_checkpoint = torch.load(file_name, map_location="cpu")
            if "best_model_state_dict" in prev_checkpoint:
                self.agent.model.load_state_dict(prev_checkpoint["best_model_state_dict"])
                logging.info("Loaded best model from {}".format(file_name))

    def _my_numel(self, m: torch.nn.Module, only_trainable: bool = False, verbose = True):
        """
        returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        model_total_params =  sum(p.numel() for p in unique)
        if verbose and self.agent.accelerator.is_main_process:
            self.agent.logger.info("Total number of trainable parameters are: {}".format(model_total_params))


        return model_total_params

    def get_scheduler(self):
        if self.agent.config.scheduler.type == "cyclic":
            after_scheduler = optim.lr_scheduler.CyclicLR(self.agent.optimizer, base_lr=self.agent.config.optimizer.learning_rate, max_lr=self.agent.config.scheduler.max_lr, cycle_momentum=False)

            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)

        elif self.agent.config.scheduler.type == "cosanneal":

            after_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.agent.optimizer, T_0=4, T_mult=2)
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)

        elif self.agent.config.scheduler.type == "reducerlonplatau":

            after_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.agent.optimizer,
                                                                   mode='min',
                                                                   factor=self.agent.config.scheduler.factor,
                                                                   patience=self.agent.config.scheduler.patience,
                                                                   verbose=True
                                                                   )
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)
        elif self.agent.config.scheduler.type == "stepLR":
            after_scheduler = optim.lr_scheduler.StepLR(optimizer=self.agent.optimizer, step_size=self.agent.config.scheduler.lr_decay_step, gamma=self.agent.config.scheduler.lr_decay_ratio)
            self.agent.scheduler = WarmupScheduler(optimizer=self.agent.optimizer,
                                                   base_lr=self.agent.config.optimizer.learning_rate,
                                                   n_warmup_steps=self.agent.config.scheduler.warm_up_steps,
                                                   after_scheduler=after_scheduler)

        else:
            self.agent.scheduler = No_Scheduler(base_lr=self.agent.config.optimizer.learning_rate)

    def sleep_load(self):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """


        if "save_base_dir" in self.agent.config.model:
            file_name = os.path.join(self.agent.config.model.save_base_dir, self.agent.config.model.save_dir)
        else:
            file_name = self.agent.config.model.save_dir

        if self.agent.accelerator.is_main_process:
            self.agent.logger.info("Loading checkpoint: {}".format(file_name))

        checkpoint = torch.load(file_name, map_location="cpu")

        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if "training_dataloder_generator_state" in checkpoint:
            self.agent.data_loader.train_loader.generator.set_state(checkpoint["training_dataloder_generator_state"])
        if "logs" in checkpoint:
            self.agent.logs = checkpoint["logs"]

        if hasattr(checkpoint, "metrics"):
            self.agent.data_loader.load_metrics_ongoing(checkpoint["metrics"])
        if hasattr(self.agent.logs, "weights"):
            self.agent.data_loader.weights = self.agent.logs["weights"]
            self.agent.weights = self.agent.data_loader.weights
            self.agent.logger.info("Loaded loss weights are:", self.agent.weights)


        for step in self.agent.logs["train_logs"]:
            wandb.log({"train": self.agent.logs["train_logs"][step], "val":  self.agent.logs["val_logs"][step]}, step=step)
            for i, lr in enumerate(self.agent.logs["train_logs"][step]["learning_rate"]):
                wandb.log({"lr": lr, "val":  self.agent.logs["val_logs"][step]}, step=i+ step - self.agent.config.early_stopping.validate_every)

        self.agent.loss = nn.CrossEntropyLoss()

        message = ""
        if "step" in self.agent.logs["best_logs"]:
            message += Fore.WHITE + "The best in step: {} so far \n".format(
                int(self.agent.logs["best_logs"]["step"] / self.agent.config.early_stopping.validate_every))

            if "loss" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["loss"].items(): message += Fore.RED + "{} : {:.6f} ".format(i,v)
            if "acc" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["acc"].items(): message += Fore.LIGHTBLUE_EX + "Acc_{}: {:.2f} ".format(i, v * 100)
            if "f1" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["f1"].items(): message += Fore.LIGHTGREEN_EX + "F1_{}: {:.2f} ".format(i, v * 100)
            if "k" in self.agent.logs["best_logs"]:
                for i, v in self.agent.logs["best_logs"]["k"].items(): message += Fore.LIGHTGREEN_EX + "K_{}: {:.4f} ".format(i, v)

        if self.agent.accelerator.is_main_process:
            self.agent.logger.info("Checkpoint loaded successfully")
            self.agent.logger.info(message)


    def sleep_load_encoder(self, enc_args):
        encs = []
        for num_enc in range(len(enc_args)):
            enc_class = globals()[enc_args[num_enc]["model"]]
            args = enc_args[num_enc]["args"]
            # print(enc_class)
            if "encoders" in enc_args[num_enc]:
                enc_enc = self.sleep_load_encoder(enc_args[num_enc]["encoders"])
                enc = enc_class(encs = enc_enc, args = args)
            else:
                enc = enc_class(args = args, encs=[])
            # enc = nn.DataParallel(enc, device_ids=[torch.device(i) for i in self.agent.config.training_params.gpu_device])
            pretrained_encoder_args =  enc_args[num_enc].get("pretrainedEncoder", {"use":False})
            if pretrained_encoder_args["use"]:
                # print("Loading encoder from {}".format(enc_args[num_enc]["pretrainedEncoder"]["dir"]))
                file_path = pretrained_encoder_args.get("dir","")
                if "save_base_dir" in self.agent.config.model:
                    file_path = os.path.join(self.agent.config.model.save_base_dir, file_path)
                checkpoint = torch.load(file_path)
                if "encoder_state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["encoder_state_dict"])
                elif "model_state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["best_model_state_dict"])

            encs.append(enc)
        return encs