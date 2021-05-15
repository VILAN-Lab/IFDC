import torch
import misc.utils as utils
from misc.rewards import get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.struc_crit = utils.StructureLosses(opt)

    def forward(self, semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, labels, masks, gts, gt_indices,
                sc_flag, struc_flag):
        opt = self.opt
        
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, labels), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(att_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')

                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
            out['cider'] = struc_loss['cider']
        elif not sc_flag:
            loss = self.crit(self.model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, labels), labels[..., 1:], masks[..., 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss

        return out
