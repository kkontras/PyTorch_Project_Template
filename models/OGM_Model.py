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


class AClassifier_VaVL_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(AClassifier_VaVL_linearcls, self).__init__()


        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        real_model_name = "wav2vec2-large-robust"
        # self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name, cache_dir="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/huggingface")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/" + real_model_name)
        self.wav2vec_model.freeze_feature_encoder()
        if real_model_name == "wav2vec2-large-robust":
            del self.wav2vec_model.encoder.layers[12:]
        # self.wav2vec_model.requires_grad_(False)

        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        self.conv_1d_a = nn.Conv1d(self.a_dim, self.d_v, kernel_size=1, padding=0, bias=False)


        self.audio_net = Conformer(
                            input_dim=self.d_v,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=5)

        # feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")

        # c = HubertConfig()
        # c.classifier_proj_size = 512
        # c.num_labels = 28
        # self.audio_net = HubertForSequenceClassification(config=c)
        # self.audio_net = self.audio_net.from_pretrained("superb/hubert-base-superb-ks")
        # self.audio_net = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")
        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
            # nn.Linear(self.d_v, num_classes)
        )


    def forward(self, x, **kwargs):

        # print(x[2].shape)
        # self.wav2vec_model.eval()
        # with torch.no_grad():
        #     x_in = self.wav2vec_model(x[2], attention_mask=None).last_hidden_state
        if "attention_mask_audio" in x:
            x_in = self.wav2vec_model(x[2], attention_mask=x["attention_mask_audio"]).last_hidden_state
        else:
            x_in = self.wav2vec_model(x[2]).last_hidden_state

        x_in = x_in.transpose(1, 2)

        # # 1-D Convolution visual/audio features
        audio = x_in if self.a_dim == self.d_v else self.conv_1d_a(x_in)
        #
        feat_a = audio.permute(2, 0, 1)
        #
        audio_feat = self.audio_net(feat_a)
        # # print(feat_a.shape)
        #
        feat_a = nn.AdaptiveAvgPool1d(1)(audio_feat.permute(1, 2, 0)).squeeze(2)
        # feat_a = nn.AdaptiveAvgPool1d(1)(x_in).squeeze(2)
        #
        pred_a = self.vclassifier(feat_a)


        # return {"preds": {"combined": pred_a}}
        return {"preds": {"combined": pred_a}, "features": {"combined": feat_a}, "nonaggr_features": {"combined": audio_feat}}

class VClassifier_FacesVaVL_linearcls(nn.Module):
    def __init__(self, args, encs):
        super(VClassifier_FacesVaVL_linearcls, self).__init__()


        self.args = args
        num_classes = args.num_classes
        d_model = args.d_model

        self.a_dim, self.v_dim = 1024, 1408
        self.d_v = 50
        self.hidden_2 = 512

        # 1D convolutional projection layers
        self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)


        self.faces_net = Conformer(
                            input_dim=self.d_v,
                            encoder_dim=self.hidden_2,
                            num_encoder_layers=5)


        self.vclassifier =  nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, **kwargs):


        x_vid = x[3].transpose(1, 2)

        # 1-D Convolution visual/audio features
        visual = x_vid if self.v_dim == self.d_v else self.conv_1d_v(x_vid)

        proj_x_v = visual.permute(2, 0, 1)
        visual_feats = self.faces_net(proj_x_v)

        feat_v = nn.AdaptiveAvgPool1d(1)(visual_feats.permute(1, 2, 0)).squeeze(2)
        # feat_a = nn.AdaptiveAvgPool1d(1)(x_in).squeeze(2)
        #
        pred_v = self.vclassifier(feat_v)



        # return {"preds": {"combined": pred_a}}
        return {"preds": {"combined": pred_v}, "features": {"combined": feat_v}, "nonaggr_features": {"combined": visual_feats}}
