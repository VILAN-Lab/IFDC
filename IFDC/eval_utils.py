from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import os
import misc.utils as utils

bad_endings = []

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 1)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration

    # Make sure in the evaluation mode
    # model = model.module
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = [] # when sample_n > 1
    # pre_file = ".json"
    # f = open(pre_file, 'w', encoding='utf-8')
    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        if data.get('labels', None) is not None and verbose_loss:
            tmp = [data['semantic_feat'], data["semantic1_feat"], data['att_feats'], data["att1_feats"], data["box_feat"], data["box1_feat"], data['labels'], data['masks']]
            tmp = [_.cuda() if _ is not None else _ for _ in tmp]
            semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, labels, masks = tmp

            with torch.no_grad():
                loss = crit(model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, labels), labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        tmp = [data['semantic_feat'],
               data['semantic1_feat'],
               data['att_feats'],
               data['att1_feats'],
               data['box_feat'],
               data['box1_feat']]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat = tmp

        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs = model(semantic_feat, semantic1_feat, att_feats, att1_feats, box_feat, box1_feat, opt=tmp_eval_kwargs, mode='sample')   ###!!!修改!!!(已修改)
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).float().sum(1)+1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).float().sum(1)+1)

        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': str(data['infos'][k]['id']), 'caption': [sent], 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_path_default'] = data['infos'][k]['file_path_default']
                entry['file_path_semantic'] = data['infos'][k]['file_path_semantic']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path_default']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                os.system(cmd)

            # if verbose:
                # example = {"id": entry['image_id'], "pre_sent": entry['caption'], "perplexity": entry['perplexity'], "entropy": entry["entropy"]}
                # f.write(json.dumps(example) + '\n')

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+ eval_kwargs['id'] + '_' + split + '.pth'))

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats