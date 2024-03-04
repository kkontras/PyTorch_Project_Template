
from datasets import *
from datasets.Mnist import *
from datasets.Color_Mnist.Color_Mnist_Dataset import *
from datasets.AVE.AVE_Dataset import *
from datasets.MOSEI.MOSEI_Dataset import *
from datasets.AVMnist.AVMnist_Dataset import *
from datasets.CREMAD.CREMAD_Dataset import *
from datasets.Kinetics.MyKinetics_Dataset import *
from datasets.Kinetics.Kinetics_Dataset import *
from datasets.UCF101.UCF101_Dataset import *
from datasets.SthSth.dataset_factory import SthSth_VideoLayoutFlow
from datasets.SthSth.Pre_Features_Dataloader import SthSth_PreFeatureDataloader

from utils.deterministic_pytorch import deterministic
from utils.misc import print_cuda_statistics
from agents.general_agent.helpers.Loader import Loader
from agents.general_agent.helpers.Monitor_n_Save import Monitor_n_Save
from agents.general_agent.helpers.Trainer import Trainer
from agents.general_agent.helpers.Validator_Tester import Validator_Tester
from agents.general_agent.helpers.Bias_Infusion import pick_bias_infuser
from agents.general_agent.helpers.Evaluator import All_Evaluator
import wandb
import torch.nn as nn
import logging
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets.MOSEI.multibenchmark_mosei_dataloader import CMU_MOSEI_Dataloader
os.environ['WANDB_SILENT'] = 'true'

class Agent():
    def __init__(self, config):
        self.config = config

        self.accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            cpu=False,
        )

        deterministic(self.config.training_params.seed)

        if self.accelerator.is_main_process: print_cuda_statistics()

        dataloader = globals()[self.config.dataset.dataloader_class]
        self.data_loader = dataloader(config=config)

        self.initialize_logs()
        self.get_loss()

        self.mem_loader = Loader(agent = self)
        self.monitor_n_saver = Monitor_n_Save(agent = self)
        self.trainer = Trainer(agent = self)
        self.validator_tester = Validator_Tester(agent = self)
        self.bias_infuser = pick_bias_infuser(agent = self)
        self.evaluators = All_Evaluator(self.config, dataloaders=self.data_loader)

        self.mem_loader.load_models_n_optimizer()
        self.mem_loader.get_scheduler()

        wandb.watch(self.model, log_freq=100)

    def initialize_logs(self):
        self.logger = logging.getLogger('Agent')
        self.logger.setLevel(logging.INFO)

        self.device = "cuda:{}".format(self.config.training_params.gpu_device[0])
        if self.accelerator.is_main_process: self.logger.info("Device: {}".format(self.device))

        self.steps_no_improve = 0
        if self.config.early_stopping.validate_every and self.config.early_stopping.end_of_epoch_check:
            max_steps = int(len(self.data_loader.train_loader) / self.config.early_stopping.validate_every) + 1

            if self.accelerator.is_main_process:
                #log that totally we have those training batches, we validate every validate_every batches and the steps per epoch are the max_steps
                self.logger.info("Total training batches: {}, validate every {} batches, steps per epoch: {}".format(
                    len(self.data_loader.train_loader), self.config.early_stopping.validate_every, max_steps))


        if "weights" not in vars(self).keys(): self.weights = None

        self.logs = {"current_epoch":0,"current_step":0,"steps_no_improve":0, "saved_step": 0, "train_logs":{},"val_logs":{},"test_logs":{},"best_logs":{"loss":{"total":100}, "acc":{"combined":0}} , "seed":self.config.training_params.seed, "weights": self.weights}
        if self.config.training_params.wandb_disable:
            self.wandb_run = wandb.init(reinit=True, project="balance", config=self.config, mode = "disabled", dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb", name= self.config.model.save_dir.split("/")[-1][:-8])
        else:
            self.wandb_run = wandb.init(reinit=True, project="balance", config=self.config, dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2021_data/wandb", name= self.config.model.save_dir.split("/")[-1][:-8] )

    def get_loss(self):

        self.loss = nn.CrossEntropyLoss()

    def accelerate_components(self):
        self.model, self.optimizer, self.data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.data_loader, self.scheduler
        )

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.model.load_ongoing:
                self.mem_loader.sleep_load()

            self.accelerate_components()
                # self.validator_tester.sleep_test()

            self.trainer.train_steps()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")
            return

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("We are in the final state.")

        self.mem_loader.load_best_model()
        # self.best_model = self.accelerator.prepare(self.best_model)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.validator_tester.validate(test_set=False)
            best_val_metrics = self.evaluators.val_evaluator.evaluate()
            self.monitor_n_saver.print_valid_results(best_val_metrics, -1)

            if hasattr(self.data_loader, "test_loader"):
                self.validator_tester.validate(test_set=True)
                best_test_metrics = self.evaluators.test_evaluator.evaluate()
                self.monitor_n_saver.print_valid_results(best_test_metrics, -1, test=True)

                if self.logs["best_logs"]["loss"]["total"] == 100:
                    self.logs["best_logs"] = best_val_metrics
                self.monitor_n_saver.sleep_save(model_save=False, verbose=True, post_test_results=best_test_metrics)
            else:
                self.monitor_n_saver.sleep_save(model_save=False, verbose=True, post_test_results=best_val_metrics)


        return self.logs["best_logs"]["loss"]["total"]

