import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class Encoder_Type_Model(nn.Module):
    def __init__(self, args, encs):
        super(Encoder_Type_Model, self).__init__()

        self.args = args

        num_classes = args.num_classes

        self.enc_0 = encs[0]
        self.enc_1 = encs[1]


        self.common_fc = nn.Sequential(
                nn.Linear(128, num_classes)
            )

    def _get_features(self, x):

        a = self.enc_0(x)
        v = self.enc_1(x)

        return a["features"]["combined"], v["features"]["combined"], a["preds"]["combined"], v["preds"]["combined"]

    def forward(self, x, **kwargs):

        feat_a, pred_a = self.enc_0(x)
        feat_v, pred_v = self.enc_1(x)
        pred = self.common_fc(torch.concatenate([feat_a,feat_v]))

        return {"preds":{"combined":pred},
                "features": {"c": feat_a, "g": feat_v}}
