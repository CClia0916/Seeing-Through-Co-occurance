# train_charades_norank1_cooccur_ablation_savebest_notext_freq.py

import time
import argparse
import csv
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

from utils import *
from apmeter import APMeter
from sentence_transformers import SentenceTransformer

try:
    from thop import profile
except ImportError:
    profile = None

parser = argparse.ArgumentParser()

"""
########
    -dataset charades
    -mode rgb
    -model MultiDimensionalDecoupling
    -train True
    -num_clips 256
    -skip 0
    -lr 0.0001
    -comp_info False
    -epoch 50
    -unisize True
    -alpha_l 1
    -batch_size 32
"""
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow (or joint for eval)')
parser.add_argument('-model', type=str, default='MultiDimensionalDecoupling',
                    choices=['MultiDimensionalDecoupling'])
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-num_clips', type=str, default='256')
parser.add_argument('-skip', type=str, default='0')
parser.add_argument('-lr', type=str, default='0.0001')
parser.add_argument('-comp_info', type=str, default='False')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-unisize', type=str, default='True')
parser.add_argument('-alpha_l', type=float, default=1.0)
parser.add_argument('-batch_size', type=str, default='32')

# ===== text-guided auxiliary hyperparams =====
parser.add_argument('-lambda_rel', type=float, default=0.1)
parser.add_argument('-corr_beta', type=float, default=0.5)
parser.add_argument('-use_txt_ctx', type=str2bool, default='False')
parser.add_argument('-cooccur_ablation', type=str, default='full',
                    choices=['full', 'no_prior', 'no_uncertainty', 'no_relation', 'no_aux'])

parser.add_argument('-freq_ablation', type=str, default='all',
                    choices=['all', 'low_only', 'mid_only', 'high_only', 'no_low', 'no_mid', 'no_high'])

"""
########
"""
parser.add_argument('-gpu', type=str, default='2')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-num_layer', type=str, default='False')

parser.add_argument('-save_root', type=str, default='./outputs/charades')
parser.add_argument('-save_best_only', type=str2bool, default='True')
parser.add_argument('-save_every_epoch', type=str2bool, default='False')
parser.add_argument('-save_val_data', type=str2bool, default='True')
parser.add_argument('-save_latest_checkpoint', type=str2bool, default='True')

parser.add_argument('-semantic_backbone', type=str, default='no_text',
                    choices=['no_text', 'full'])
parser.add_argument('-annotation_path', type=str, default='./data/charades.json')
parser.add_argument('-text_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
parser.add_argument('-text_cache', type=str, default='./cache/charades_text_embeddings.pt')
parser.add_argument('-cooccur_cache', type=str, default='./cache/charades_cooccurrence.pt')
parser.add_argument('-num_workers', type=int, default=8)
parser.add_argument('-profile_model', type=str2bool, default='False')

args = parser.parse_args()

if __name__ == '__main__':
    selected_root = args.rgb_root if args.mode == 'rgb' else args.flow_root
    if selected_root == 'no_root':
        raise ValueError(f"Set -{args.mode}_root to the directory containing <video_id>.npy features.")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
SAVE_ROOT = os.path.join(
    args.save_root,
    f"semantic_{args.semantic_backbone}_cooccur_{args.cooccur_ablation}_freq_{args.freq_ablation}"
)
PRED_DIR = os.path.join(SAVE_ROOT, 'preds')
CKPT_DIR = os.path.join(SAVE_ROOT, 'checkpoints')
LOG_DIR = os.path.join(SAVE_ROOT, 'logs')
FIG_DIR = os.path.join(SAVE_ROOT, 'figures')
for _d in [SAVE_ROOT, PRED_DIR, CKPT_DIR, LOG_DIR, FIG_DIR]:
    os.makedirs(_d, exist_ok=True)

# =========================================================
# Random seed
# =========================================================
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED:', SEED)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=3, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


newloss = AsymmetricLoss()
batch_size = int(args.batch_size)

# =========================================================
# Dataset
# =========================================================
if args.dataset == 'charades':
    from datasets.charades import Charades as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to", args.num_clips)
        from datasets.charades import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from datasets.charades import mt_collate_fn as collate_fn

    train_split = args.annotation_path
    test_split = train_split
    rgb_root = args.rgb_root
    flow_root = args.flow_root
    classes = 157

# =========================================================
# Charades 157 phrases
# =========================================================
class_labels = """
c000 Holding some clothes
c001 Putting clothes somewhere
c002 Taking clothes from somewhere
c003 Throwing clothes somewhere
c004 Tidying some clothes
c005 Washing some clothes
c006 Closing a door
c007 Fixing a door
c008 Opening a door
c009 Putting something on a table
c010 Sitting on a table
c011 Sitting at a table
c012 Tidying up a table
c013 Washing a table
c014 Working at a table
c015 Holding a phone/camera
c016 Playing with a phone/camera
c017 Putting a phone/camera somewhere
c018 Taking a phone/camera from somewhere
c019 Talking on a phone/camera
c020 Holding a bag
c021 Opening a bag
c022 Putting a bag somewhere
c023 Taking a bag from somewhere
c024 Throwing a bag somewhere
c025 Closing a book
c026 Holding a book
c027 Opening a book
c028 Putting a book somewhere
c029 Smiling at a book
c030 Taking a book from somewhere
c031 Throwing a book somewhere
c032 Watching/Reading/Looking at a book
c033 Holding a towel/s
c034 Putting a towel/s somewhere
c035 Taking a towel/s from somewhere
c036 Throwing a towel/s somewhere
c037 Tidying up a towel/s
c038 Washing something with a towel
c039 Closing a box
c040 Holding a box
c041 Opening a box
c042 Putting a box somewhere
c043 Taking a box from somewhere
c044 Taking something from a box
c045 Throwing a box somewhere
c046 Closing a laptop
c047 Holding a laptop
c048 Opening a laptop
c049 Putting a laptop somewhere
c050 Taking a laptop from somewhere
c051 Watching a laptop or something on a laptop
c052 Working/Playing on a laptop
c053 Holding a shoe/shoes
c054 Putting shoes somewhere
c055 Putting on shoe/shoes
c056 Taking shoes from somewhere
c057 Taking off some shoes
c058 Throwing shoes somewhere
c059 Sitting in a chair
c060 Standing on a chair
c061 Holding some food
c062 Putting some food somewhere
c063 Taking food from somewhere
c064 Throwing food somewhere
c065 Eating a sandwich
c066 Making a sandwich
c067 Holding a sandwich
c068 Putting a sandwich somewhere
c069 Taking a sandwich from somewhere
c070 Holding a blanket
c071 Putting a blanket somewhere
c072 Snuggling with a blanket
c073 Taking a blanket from somewhere
c074 Throwing a blanket somewhere
c075 Tidying up a blanket/s
c076 Holding a pillow
c077 Putting a pillow somewhere
c078 Snuggling with a pillow
c079 Taking a pillow from somewhere
c080 Throwing a pillow somewhere
c081 Putting something on a shelf
c082 Tidying a shelf or something on a shelf
c083 Reaching for and grabbing a picture
c084 Holding a picture
c085 Laughing at a picture
c086 Putting a picture somewhere
c087 Taking a picture of something
c088 Watching/looking at a picture
c089 Closing a window
c090 Opening a window
c091 Washing a window
c092 Watching/Looking outside of a window
c093 Holding a mirror
c094 Smiling in a mirror
c095 Washing a mirror
c096 Watching something/someone/themselves in a mirror
c097 Walking through a doorway
c098 Holding a broom
c099 Putting a broom somewhere
c100 Taking a broom from somewhere
c101 Throwing a broom somewhere
c102 Tidying up with a broom
c103 Fixing a light
c104 Turning on a light
c105 Turning off a light
c106 Drinking from a cup/glass/bottle
c107 Holding a cup/glass/bottle of something
c108 Pouring something into a cup/glass/bottle
c109 Putting a cup/glass/bottle somewhere
c110 Taking a cup/glass/bottle from somewhere
c111 Washing a cup/glass/bottle
c112 Closing a closet/cabinet
c113 Opening a closet/cabinet
c114 Tidying up a closet/cabinet
c115 Someone is holding a paper/notebook
c116 Putting their paper/notebook somewhere
c117 Taking paper/notebook from somewhere
c118 Holding a dish
c119 Putting a dish/es somewhere
c120 Taking a dish/es from somewhere
c121 Wash a dish/dishes
c122 Lying on a sofa/couch
c123 Sitting on sofa/couch
c124 Lying on the floor
c125 Sitting on the floor
c126 Throwing something on the floor
c127 Tidying something on the floor
c128 Holding some medicine
c129 Taking/consuming some medicine
c130 Putting groceries somewhere
c131 Laughing at television
c132 Watching television
c133 Someone is awakening in bed
c134 Lying on a bed
c135 Sitting in a bed
c136 Fixing a vacuum
c137 Holding a vacuum
c138 Taking a vacuum from somewhere
c139 Washing their hands
c140 Fixing a doorknob
c141 Grasping onto a doorknob
c142 Closing a refrigerator
c143 Opening a refrigerator
c144 Fixing their hair
c145 Working on paper/notebook
c146 Someone is awakening somewhere
c147 Someone is cooking something
c148 Someone is dressing
c149 Someone is laughing
c150 Someone is running somewhere
c151 Someone is going from standing to sitting
c152 Someone is smiling
c153 Someone is sneezing
c154 Someone is standing up from somewhere
c155 Someone is undressing
c156 Someone is eating something
"""
_lines = [l.strip() for l in class_labels.strip().split("\n") if l.strip()]
phrase_list = [l.split(" ", 1)[1] for l in _lines]
assert len(phrase_list) == 157, f"phrase_list len {len(phrase_list)} != 157"

# =========================================================
# Phrase-BERT encode once
# =========================================================
PHRASE_BERT_PATH = args.text_model
PHRASE_CACHE = args.text_cache
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(PHRASE_CACHE):
    phrase_embs = torch.load(PHRASE_CACHE, map_location="cpu").to(DEVICE)
else:
    _pb = SentenceTransformer(PHRASE_BERT_PATH, device=DEVICE)
    with torch.no_grad():
        phrase_embs = _pb.encode(
            phrase_list,
            convert_to_tensor=True,
            device=DEVICE,
            show_progress_bar=True
        )
    cache_dir = os.path.dirname(os.path.abspath(PHRASE_CACHE))
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(phrase_embs.detach().cpu(), PHRASE_CACHE)

# =========================================================
# Data loader
# =========================================================
def load_data(train_split, val_split, root):
    print('load data', root)

    if len(train_split) > 0:
        dataset = Dataset(
            train_split, 'training', root,
            batch_size, classes, int(args.num_clips), int(args.skip)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        dataloader.root = root
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(
        val_split, 'testing', root,
        batch_size, classes, int(args.num_clips), int(args.skip)
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, args.num_workers // 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets

# =========================================================
# Offline time-step co-occurrence prior
# =========================================================
def count_time_step_cooccurrence(dataloader, num_classes=157, save_path="./charades_time_step_cooccur.pt"):
    cooccur_count = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    total_valid_steps = 0

    for data in dataloader:
        _, mask, labels, _, _ = data
        mask = mask.bool()
        labels = labels.float()

        B, T, N = labels.shape
        for b in range(B):
            valid_t = mask[b]
            for t in range(T):
                if not valid_t[t]:
                    continue
                total_valid_steps += 1
                y = labels[b, t]
                cooccur_count += torch.ger(y, y)

    alpha = 1e-3
    class_freq = cooccur_count.diag().unsqueeze(1)
    cooccur_prob = (cooccur_count + alpha) / (class_freq + alpha * num_classes)
    cooccur_prob.fill_diagonal_(1.0)

    torch.save({
        "cooccur_count": cooccur_count,
        "cooccur_prob": cooccur_prob,
        "total_valid_steps": total_valid_steps
    }, save_path)
    print(f"Charades共现矩阵统计完成，已保存至{save_path}")
    return cooccur_count, cooccur_prob

# =========================================================
# Text-guided auxiliary modules
# =========================================================
class TextPrototypeTeacher(nn.Module):
    def __init__(self, phrase_embs: torch.Tensor, d_e: int = 512, use_context: bool = True):
        super().__init__()
        self.register_buffer("phrase_embs", phrase_embs)
        d_text = phrase_embs.size(1)
        self.proj = nn.Linear(d_text, d_e)
        self.use_context = use_context
        if use_context:
            self.ctx = nn.Sequential(
                nn.Linear(d_e, d_e),
                nn.ReLU(),
                nn.Linear(d_e, d_e),
            )
        self.norm = nn.LayerNorm(d_e)

    def forward(self, presence=None):
        P = self.norm(self.proj(self.phrase_embs))
        if (presence is None) or (not self.use_context):
            return P
        pres = presence.float()
        denom = pres.sum(dim=1, keepdim=True).clamp_min(1.0)
        c = (pres @ P) / denom
        delta = self.ctx(c).unsqueeze(1)
        return self.norm(P.unsqueeze(0) + delta)


class CooccurAwareFusionHead(nn.Module):
    def __init__(self, d_in=512, num_classes=157, txt_dim=512, init_beta=0.5):
        super().__init__()
        self.vis_cls = nn.Linear(d_in, num_classes)
        self.feat_proj = nn.Linear(d_in, txt_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, X, Ptxt, mode='full'):
        if Ptxt.dim() == 2:
            Ptxt = Ptxt.unsqueeze(0).expand(X.size(0), -1, -1)

        V_aux = self.vis_cls(X)
        Xp = F.normalize(self.feat_proj(X), dim=-1)
        Pn = F.normalize(Ptxt, dim=-1)
        T_aux = torch.einsum('btd,bnd->btn', Xp, Pn)
        T_aux = self.logit_scale.exp().clamp(max=100.0) * T_aux

        pv = torch.sigmoid(V_aux)
        uncertainty = 1.0 - torch.abs(2.0 * pv - 1.0)
        beta = self.beta.clamp(0.0, 1.0)

        if mode == 'no_uncertainty':
            F_aux = V_aux + beta * (T_aux - V_aux)
        else:
            F_aux = V_aux + beta * uncertainty * (T_aux - V_aux)

        return V_aux, T_aux, F_aux, uncertainty

# =========================================================
# Losses
# =========================================================
def pairwise_cooccurrence_loss(logits, labels, mask, cooccur_prior=None, alpha=0.5, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred_rel = prob.unsqueeze(-1) * prob.unsqueeze(-2)
    gt_rel = labels.unsqueeze(-1) * labels.unsqueeze(-2)

    if cooccur_prior is None:
        weight = 1.0
    else:
        weight = alpha + (1.0 - alpha) * cooccur_prior
        weight = weight.unsqueeze(0).unsqueeze(0)

    rel_err = torch.abs(pred_rel - gt_rel)
    if isinstance(weight, torch.Tensor):
        rel_err = rel_err * weight

    valid = mask.unsqueeze(-1).unsqueeze(-1).expand_as(rel_err)
    return (rel_err * valid).sum() / valid.sum().clamp_min(eps)


def _to_float(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().cpu().mean().item())
    return float(x)


def append_metrics_row(epoch, train_map, train_loss, val_loss, val_map, sample_val_map, csv_path):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_map', 'train_loss', 'val_loss', 'val_map', 'sample_val_map'])
        writer.writerow([
            int(epoch),
            float(train_map),
            float(train_loss),
            float(val_loss),
            float(val_map),
            float(sample_val_map)
        ])


def save_checkpoint(path, epoch, model, optimizer, sched, best_val_map, teacher=None, fusion_head=None, extra=None):
    state = {
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': sched.state_dict() if sched is not None else None,
        'best_val_map': float(best_val_map),
        'args': vars(args)
    }
    if teacher is not None:
        state['teacher_state_dict'] = teacher.state_dict()
    if fusion_head is not None:
        state['fusion_head_state_dict'] = fusion_head.state_dict()
    if extra is not None:
        state['extra'] = extra
    torch.save(state, path)


def save_prediction_bundle(path, epoch, full_probs, raw_val_data, val_map, sample_val_map):
    bundle = {
        'epoch': int(epoch),
        'val_map': float(val_map),
        'sample_val_map': float(sample_val_map),
        'full_probs': full_probs,
        'raw_val_data': raw_val_data,
        'cooccur_ablation': args.cooccur_ablation,
        'semantic_backbone': args.semantic_backbone,
        'freq_ablation': args.freq_ablation,
    }
    with open(path, 'wb') as f:
        pickle.dump(bundle, f, pickle.HIGHEST_PROTOCOL)

teacher = None
fusion_head = None
cooccur_prob = None

def run(models, criterion, num_epochs=50):
    since = time.time()
    best_val_map = -1.0
    best_epoch = -1
    metrics_csv = os.path.join(LOG_DIR, 'metrics.csv')
    summary_txt = os.path.join(LOG_DIR, 'summary.txt')

    for epoch in range(num_epochs):
        since1 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for model, gpu, dataloader, optimizer, sched, model_file in models:
            train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, raw_val_data, val_loss, val_map, sample_val_map = val_step(model, gpu, dataloader['val'], epoch)
            sched.step(val_loss)

            train_map_f = _to_float(train_map)
            train_loss_f = _to_float(train_loss)
            val_loss_f = _to_float(val_loss)
            val_map_f = _to_float(val_map)
            sample_val_map_f = _to_float(sample_val_map)

            print("epoch", epoch, "Total_Time", time.time() - since, "Epoch_time", time.time() - since1)

            append_metrics_row(epoch, train_map_f, train_loss_f, val_loss_f, val_map_f, sample_val_map_f, metrics_csv)

            if bool(args.save_latest_checkpoint):
                save_checkpoint(
                    os.path.join(CKPT_DIR, 'latest.pth'),
                    epoch, model, optimizer, sched, max(best_val_map, val_map_f),
                    teacher=teacher, fusion_head=fusion_head,
                    extra={'train_map': train_map_f, 'train_loss': train_loss_f, 'val_loss': val_loss_f,
                           'val_map': val_map_f, 'sample_val_map': sample_val_map_f}
                )

            if bool(args.save_every_epoch):
                save_checkpoint(
                    os.path.join(CKPT_DIR, f'epoch_{epoch:03d}.pth'),
                    epoch, model, optimizer, sched, max(best_val_map, val_map_f),
                    teacher=teacher, fusion_head=fusion_head,
                    extra={'train_map': train_map_f, 'train_loss': train_loss_f, 'val_loss': val_loss_f,
                           'val_map': val_map_f, 'sample_val_map': sample_val_map_f}
                )
                if bool(args.save_val_data):
                    save_prediction_bundle(
                        os.path.join(PRED_DIR, f'epoch_{epoch:03d}_val.pkl'),
                        epoch, prob_val, raw_val_data, val_map_f, sample_val_map_f
                    )

            if val_map_f > best_val_map:
                best_val_map = val_map_f
                best_epoch = epoch
                print("epoch", epoch, "Best Val Map Update", best_val_map)
                save_checkpoint(
                    os.path.join(CKPT_DIR, 'best.pth'),
                    epoch, model, optimizer, sched, best_val_map,
                    teacher=teacher, fusion_head=fusion_head,
                    extra={'train_map': train_map_f, 'train_loss': train_loss_f, 'val_loss': val_loss_f,
                           'val_map': val_map_f, 'sample_val_map': sample_val_map_f}
                )
                if bool(args.save_val_data):
                    save_prediction_bundle(
                        os.path.join(PRED_DIR, 'best_val.pkl'),
                        epoch, prob_val, raw_val_data, val_map_f, sample_val_map_f
                    )

    with open(summary_txt, 'w') as f:
        f.write('best_epoch: %s\n' % str(best_epoch))
        f.write('best_val_map: %.6f\n' % float(best_val_map))
        f.write('save_root: %s\n' % SAVE_ROOT)
        f.write('semantic_backbone: %s\n' % str(args.semantic_backbone))
        f.write('cooccur_ablation: %s\n' % str(args.cooccur_ablation))
        f.write('freq_ablation: %s\n' % str(args.freq_ablation))


def run_network(model, data, gpu, epoch=0, baseline=False, is_train=True):
    global teacher, fusion_head, cooccur_prob

    inputs, mask, labels, other, _ = data

    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu)).float()
    labels = Variable(labels.cuda(gpu)).float()

    inputs = inputs.squeeze(3).squeeze(3)
    outputs_final, out_hm = model(inputs)

    loss_main = newloss(outputs_final, labels) / torch.sum(mask)
    loss = args.alpha_l * loss_main

    if is_train and args.cooccur_ablation != 'no_aux':
        if out_hm.dim() != 3:
            raise ValueError(f"out_hm shape unexpected: {out_hm.shape}, expected 3D tensor like (B,D,T)")

        X0 = out_hm.transpose(1, 2)
        masked_labels = labels * mask.unsqueeze(-1)
        presence = (masked_labels.max(dim=1).values > 0.5).float()

        Ptxt = teacher(presence=presence if bool(args.use_txt_ctx) else None)
        if Ptxt.dim() == 2:
            Ptxt = Ptxt.unsqueeze(0).expand(X0.size(0), -1, -1)

        V_aux, T_aux, F_aux, uncertainty = fusion_head(X0, Ptxt, mode=args.cooccur_ablation)

        if args.cooccur_ablation != 'no_relation':
            use_prior = None if args.cooccur_ablation == 'no_prior' else (
                cooccur_prob.to(F_aux.device) if cooccur_prob is not None else None
            )
            loss_rel = pairwise_cooccurrence_loss(
                logits=F_aux,
                labels=labels,
                mask=mask,
                cooccur_prior=use_prior
            )
            loss = loss + float(args.lambda_rel) * loss_rel

    probs_f = torch.sigmoid(outputs_final) * mask.unsqueeze(2)
    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    global teacher, fusion_head

    model.train(True)
    if teacher is not None:
        teacher.train(True)
    if fusion_head is not None:
        fusion_head.train(True)

    tot_loss = 0.0
    num_iter = 0.
    apm = APMeter()

    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch, is_train=True)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    print('epoch', epoch, 'train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter
    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    global teacher, fusion_head

    model.train(False)
    if teacher is not None:
        teacher.train(False)
    if fusion_head is not None:
        fusion_head.train(False)

    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    num_iter = 0.
    full_probs = {}
    raw_val_data = {}

    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch, is_train=False)

        if sum(data[1].numpy()[0]) > 25:
            p1, l1 = sampled_25(probs.data.cpu().numpy()[0], data[2].numpy()[0], data[1].numpy()[0])
            sampled_apm.add(p1, l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        tot_loss += loss.data

        probs_1 = mask_probs(probs.data.cpu().numpy()[0], data[1].numpy()[0]).squeeze()
        vid = other[0][0]
        full_probs[vid] = probs_1.T
        raw_val_data[vid] = {
            'logits': outputs.detach().cpu().numpy()[0].astype(np.float32),
            'probs': probs.detach().cpu().numpy()[0].astype(np.float32),
            'labels': data[2].numpy()[0].astype(np.float32),
            'mask': data[1].numpy()[0].astype(np.float32),
            'other': other,
        }

    epoch_loss = tot_loss / num_iter
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    sample_val_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    print('epoch', epoch, 'Full-val-map:', val_map)
    print('epoch', epoch, 'sampled-val-map:', sample_val_map)

    sampled_vec = (100 * sampled_apm.value()).detach().cpu().numpy() \
        if torch.is_tensor(100 * sampled_apm.value()) else 100 * sampled_apm.value()

    apm.reset()
    sampled_apm.reset()

    with open(os.path.join(LOG_DIR, 'log.txt'), 'a') as f:
        f.write("epoch: " + str(epoch) +
                ", Full-val-map: " + str(val_map) +
                ", sampled-val-map: " + str(sample_val_map) +
                ", epoch_loss: " + str(epoch_loss) +
                ", semantic_backbone: " + str(args.semantic_backbone) +
                ", cooccur_ablation: " + str(args.cooccur_ablation) +
                ", freq_ablation: " + str(args.freq_ablation) + "\n")
        f.write("sampled_apm: " + str(sampled_vec) + "\n")

    return full_probs, raw_val_data, epoch_loss, val_map, sample_val_map


if __name__ == '__main__':

    if args.mode == 'flow':
        if args.flow_root == 'no_root':
            raise ValueError("Set -flow_root to the directory containing <video_id>.npy features.")
        print('flow mode', args.flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'rgb':
        if args.rgb_root == 'no_root':
            raise ValueError("Set -rgb_root to the directory containing <video_id>.npy features.")
        print('RGB mode', args.rgb_root)
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)
    else:
        raise ValueError("args.mode should be 'rgb' or 'flow'")

    COOCCUR_PATH = args.cooccur_cache
    os.makedirs(os.path.dirname(os.path.abspath(COOCCUR_PATH)), exist_ok=True)
    if not os.path.exists(COOCCUR_PATH):
        print("Charades matrix is no exist")
        _, cooccur_prob = count_time_step_cooccurrence(
            dataloaders['train'], num_classes=classes, save_path=COOCCUR_PATH
        )
    else:
        print("Loading existing Charades matrix...")
        cooccur_data = torch.load(COOCCUR_PATH, map_location='cpu')
        cooccur_prob = cooccur_data["cooccur_prob"]

    if args.train:
        if args.model == "MultiDimensionalDecoupling":
            print("MultiDimensionalDecoupling")
            from model.multidimensional_decoupling import MultiDimensionalDecoupling

            num_classes = classes
            inter_channels = [256, 384, 576, 864]
            num_block = [1, 3, 3, 3]
            head = 8
            mlp_ratio = 8
            in_feat_dim = 768
            final_embedding_dim = 512

            fine_ratio = 8
            head_token = 4
            num_clips = int(args.num_clips)
            inter_tokens = [
                num_clips // fine_ratio,
                num_clips // (fine_ratio * 2),
                num_clips // (fine_ratio * 4),
                num_clips // (fine_ratio * 8),
            ]
            inter_channels_slow = [256, 384, 576, 864]
            k = 4
            num_block_spatial = [1, 1, 3, 1]

            rgb_model = MultiDimensionalDecoupling(
                inter_channels, inter_channels_slow, num_block, num_block_spatial, head, mlp_ratio,
                in_feat_dim, final_embedding_dim, num_classes, fine_ratio, inter_tokens,
                head_token, k
            )
            rgb_model.set_text_prototypes(phrase_embs)
            rgb_model.set_semantic_ablation(args.semantic_backbone)
            rgb_model.set_freq_ablation(args.freq_ablation)

            print("semantic backbone =", args.semantic_backbone)
            print("freq ablation =", args.freq_ablation)
            print("loaded", args.load_model)

            if bool(args.profile_model):
                if profile is None:
                    raise ImportError("Install thop to use -profile_model True.")
                input_tensor = torch.randn(1, in_feat_dim, num_clips)
                flops, params = profile(rgb_model, (input_tensor,))
                print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                print('Params = ' + str(params / 1000 ** 2) + 'M')

            if args.cooccur_ablation != 'no_aux':
                teacher = TextPrototypeTeacher(
                    phrase_embs=phrase_embs,
                    d_e=final_embedding_dim,
                    use_context=bool(args.use_txt_ctx)
                )

                fusion_head = CooccurAwareFusionHead(
                    d_in=final_embedding_dim,
                    num_classes=classes,
                    txt_dim=final_embedding_dim,
                    init_beta=float(args.corr_beta)
                )
            else:
                teacher = None
                fusion_head = None

        rgb_model.cuda()
        if teacher is not None:
            teacher.cuda()
        if fusion_head is not None:
            fusion_head.cuda()

        param_groups = list(rgb_model.parameters())
        if teacher is not None:
            param_groups += list(teacher.parameters())
        if fusion_head is not None:
            param_groups += list(fusion_head.parameters())

        optimizer = optim.AdamW(param_groups, lr=float(args.lr))

        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=2,
            verbose=True
        )

        run(
            [(rgb_model, 0, dataloaders, optimizer, lr_sched, args.comp_info)],
            None,
            num_epochs=int(args.epoch)
        )
