import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score
from collections import defaultdict

class Validator_Tester():
    def __init__(self, agent):
        self.agent = agent
        self.multi_supervised = False

        self.valtest_step_func = "valtest_one_step"

        self.this_valtest_step_func = getattr(self, self.valtest_step_func)
        self._get_loss_weights()

    def validate(self, best_model = False, test_set= False):


        self.agent.model.eval()
        self.agent.model.train(False)

        this_evaluator = self.agent.evaluators.test_evaluator if test_set else self.agent.evaluators.val_evaluator
        this_dataloader = self.agent.data_loader.test_loader if test_set else self.agent.data_loader.valid_loader
        this_evaluator.reset()
        with torch.no_grad():
            pbar = tqdm(enumerate(this_dataloader),
                        total=len(this_dataloader),
                        desc="Validation",
                        leave=False,
                        disable=True,
                        position=1)
            for batch_idx, served_dict in pbar:

                step_outcome = self.this_valtest_step_func(served_dict, best_model=best_model)

                all_outputs = self.agent.accelerator.gather(step_outcome)
                this_evaluator.process(all_outputs)

                del step_outcome, served_dict

                mean_batch_loss, mean_batch_loss_message = this_evaluator.mean_batch_loss()

                pbar_message = "Validation batch {0:d}/{1:d} with {2:}".format(batch_idx,
                                                                             len(this_dataloader) - 1,
                                                                             mean_batch_loss_message)
                pbar.set_description(pbar_message)
                pbar.refresh()

    def valtest_one_step(self, served_dict, best_model=False):

            data = {view: served_dict["data"][view].cuda() for view in
                                   served_dict["data"] if type(served_dict["data"][view]) is torch.Tensor }
            data.update({view: data[view].float() for view in data if type(view) == int})

            label = served_dict["label"].squeeze().type(torch.LongTensor).cuda()

            output = self.agent.model(data)

            total_loss =  torch.zeros(1).squeeze().cuda()
            output_losses, ce_loss = {}, {}

            if self.agent.config.model.args.get("validation_loss_w", {"combined":1}):
                for k, v in output["preds"].items():
                    if k in self.agent.config.model.args.multi_loss.multi_supervised_w and self.agent.config.model.args.multi_loss.multi_supervised_w[k] != 0:
                        if len(label) > 0:  # TODO: Check if this one needs to be one or zero
                            ce_loss[k] = self.agent.loss(v, label)
                            total_loss += self.w_loss[k] * ce_loss[k]
                            ce_loss[k] = ce_loss[k].detach()
                            output_losses.update({"ce_loss_{}".format(k): ce_loss[k]})

            total_loss = total_loss.detach()
            output_losses.update({"total": total_loss})
            for i in output["preds"]:  output["preds"][i] =  output["preds"][i].detach()

            return {"loss": output_losses,
                   "pred" : output["preds"],
                   "label": label}


    def _get_loss_weights(self):

        w_loss = defaultdict(int)
        w_loss["total"] = 1

        ws = self.agent.config.model.args.get("validation_loss_w", {"combined":1})
        for k, v in ws.items():
            w_loss[k] = v

        self.w_loss = w_loss


