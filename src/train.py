import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import os
import json
import numpy as np
from collections import defaultdict
import random
from tqdm.notebook import tqdm
from torchvision import models
from torch.nn.utils.rnn import pad_sequence
import matplotlib.patches as patches

import math
import time
import os
from PIL import Image
import requests
import nltk

import os
import cv2
import colorsys
from numpy import asarray
import math


from transformers import GPT2LMHeadModel, GPT2Config

from transformers import BertTokenizer


from scipy.optimize import linear_sum_assignment




class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, instance_file, max_objects=40, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_objects = max_objects
        self.img_cache = dict()  # Cache for images
        
        # Load instance file only once
        with open(instance_file, 'r') as file:
            data = json.load(file)
            instances = data['annotations']
            categories = data['categories']

        with open(annotation_file, 'r') as file:
            annotations = json.load(file)['annotations']
            
        self.image_captions = defaultdict(list)
        for annotation in annotations:
            img_id = annotation['image_id']
            self.image_captions[img_id].append(annotation['caption'])

        self.image_instances = defaultdict(list)
        self.category_id_to_name = {category['id']: category['name'] for category in categories}
        
        for instance in instances:
            img_id = instance['image_id']
            bbox = instance['bbox']
            category_id = instance['category_id']
            self.image_instances[img_id].append((bbox, category_id))

        self.img_ids = list(self.image_captions.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_path = os.path.join(self.root_dir, f'{str(img_id).zfill(12)}.jpg')
        
        # Use cached image if available
        
        if img_id in self.img_cache:
            img = self.img_cache[img_id]
        else:
            img = Image.open(img_path).convert("RGB")
            self.img_cache[img_id] = img
        

        captions = self.image_captions[img_id]
        caption = random.choice(captions)

        annotations = self.image_instances[img_id]
        bboxes = []
        labels = []
        for obbox, label_id in annotations:
            bbox = torch.tensor(obbox)  # Convert to PyTorch tensor immediately
            bbox[0] = bbox[0] / img.width + (bbox[2] / img.width)/2
            bbox[1] = bbox[1] / img.height + (bbox[3] / img.height)/2
            bbox[2] = bbox[2] / img.width
            bbox[3] = bbox[3] / img.height
            label = self.category_id_to_name[label_id]
            bboxes.append(bbox)
            labels.append(label)

        bboxes.append(torch.tensor([0.5, 0.5, 1, 1]))
        labels.append(caption)
        
        total_boxes = len(bboxes)
        
        if total_boxes < 40:
            for _ in range(40-total_boxes):
                bboxes.append(torch.tensor([0, 0, 0 ,0]))
                labels.append("na")
        else:
            bboxes = bboxes[:40]
            labels = labels[:40]

        if self.transform:
            img = self.transform(img)

        return img, bboxes, labels

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def custom_collate(batch):
    images, boxes_list, labels_list = zip(*batch)

    # Convert list of PIL images to a single PyTorch tensor
    stacked_images = torch.stack(images)

    # Convert list of list of boxes to a list of PyTorch tensors
    stacked_boxes = [torch.stack([box.clone().detach() for box in boxes]) for boxes in boxes_list]


    # Since labels are strings, we can keep them as a list of lists
    # labels_list is already in the desired format

    return stacked_images, stacked_boxes, labels_list








def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()
    summary_loss = AverageMeter()    
    
    tk0 = tqdm(data_loader, total=len(data_loader)-1)
    
    for step, (images, bboxes, captions) in enumerate(tk0):
    
        try:
            flattened_captions = [caption for sublist in captions for caption in sublist]
            captions = tokenizer(flattened_captions, padding=True, return_tensors="pt", truncation=True)
            captions = captions["input_ids"]
            input_ids = captions.reshape(batch_size, num_queries, -1).to(device)
            min_length = 2
        except RuntimeError as e:
            print("Reshape failed:", e)
            continue
        
        '''
        min_length = 2
        if input_ids.size(-1) < min_length:
            padding_needed = min_length - input_ids.size(-1)
            input_ids = F.pad(input_ids, (0, padding_needed), 'constant', PAD_TOKEN)

        targets = build_targets(bboxes, input_ids[:, :, 1:])        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        images = list(image.to(device) for image in images)
        
                
        output = model(images,input_ids[:, :,:-1])
        '''
        
        min_length = 2
        if input_ids.size(-1) < min_length:
            padding_needed = min_length - input_ids.size(-1)
            input_ids = F.pad(input_ids, (0, padding_needed), 'constant', PAD_TOKEN)
            
        # input_ids = captions["input_ids"]
        # input_ids = input_ids.reshape(batch_size, num_queries, -1).to(device)
            
        targets = build_targets(bboxes, input_ids[:, :, 1:])

        #targets = build_targets(bboxes, captions[:,:,1:])
        
        images = list(image.to(device) for image in images)
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

        output = model(images,input_ids[:,:,:-1])

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step() 
        
        if scheduler is not None:
            scheduler.step()
    
        # Detach and delete tensors
        loss_dict = {k: v.detach() for k, v in loss_dict.items()}
    
        del images, bboxes, captions, output, targets, loss_dict
        torch.cuda.empty_cache()  # Clear cache
        
        summary_loss.update(losses.item(),BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)
    
        
    return summary_loss
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        
        out_prob = outputs["pred_logits"].flatten(0,2 ).softmax(-1)  # [batch_size * num_queries * seq_length, vocab_size ]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class.mean() + self.cost_giou * cost_giou
        #C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, vocab_size, matcher, weight_dict, eos_coef, losses,pad_token):
        """ Create the criterion.
        Parameters:
            vocab_size : es number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.pad_token=pad_token
        empty_weight = torch.ones(self.vocab_size)
        # empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
    
        """Classification loss (NLL) for sequences
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes, seq_length]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        batch_size, num_boxes , sequence_length, _ = src_logits.size()

        # Get the indices for the permutation
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Ensure the target classes are valid
        assert (target_classes >= 0).all() and (target_classes < self.vocab_size).all(), "Invalid token index in target!"
        
        # loss_ce = criterion(outputs.reshape(-1, vocab_size), captions.view(-1))
        loss_ce = self.criterion(src_logits.reshape(batch_size * num_boxes * sequence_length, -1), target_classes.reshape(-1))
        
        
        # loss_ce = torchmetrics.functional.smooth_cross_entropy(src_logits[batch_idx], target_classes, ignore_index=PAD_TOKEN)
        losses = {'loss_ce': loss_ce}

        return losses


        

    '''
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN)
        loss_ce = criterion(src_logits, target_classes_for_loss)
        losses = {'loss_ce': loss_ce}
    '''
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        
        card_pred = card_pred.sum(dim=1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # print("indice len", len(indices), "len (indices[0])  ", len (indices[0]))
        # print( " shape indices 0   0 ", indices [0][0].shape , " shape indices 0   1 ", indices [0][1].shape)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # print("num_boxes",num_boxes)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        '''
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        '''
        return losses

def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        
        #tk0 = tqdm(data_loader, total=len(data_loader))
        #for step, (images, bboxes, captions) in enumerate(tk0):
        #pbar = tqdm(range(len(data_loader)))** 
        
        tk0 = tqdm(data_loader, total=len(data_loader)-1)
        for step, (images, bboxes, captions) in enumerate(tk0):       
        
            try:
                flattened_captions = [caption for sublist in captions for caption in sublist]
                captions = tokenizer(flattened_captions, padding=True, return_tensors="pt", truncation=True)
                captions = captions["input_ids"]
                input_ids = captions.reshape(batch_size, num_queries, -1).to(device)
                min_length = 2
            except RuntimeError as e:
                print("Reshape failed:", e)
                continue
                    
            if input_ids.size(-1) < min_length:
                padding_needed = min_length - input_ids.size(-1)
                input_ids = F.pad(input_ids, (0, padding_needed), 'constant', PAD_TOKEN)
            
            # input_ids = captions["input_ids"]
            # input_ids = input_ids.reshape(batch_size, num_queries, -1).to(device)
            
            targets = build_targets(bboxes, input_ids[:, :, 1:])

            #targets = build_targets(bboxes, captions[:,:,1:])
        
            images = list(image.to(device) for image in images)
        
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        

            output = model(images,input_ids[:,:,:-1])
        

            loss_dict = criterion(output, targets)
            weight_dict = criterion.weight_dict
        
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
            
            summary_loss.update(losses.item(),BATCH_SIZE)
            
            #
    
            # Detach and delete tensors
            loss_dict = {k: v.detach() for k, v in loss_dict.items()}
    
            del images, bboxes, captions, output, targets, loss_dict
            torch.cuda.empty_cache()  # Clear cache
            
            tk0.set_postfix(loss=summary_loss.avg)
        #data_loader.on_epoch_end()
    
    return summary_loss

def build_targets(bboxes, captions):
    targets = []
    for  i, (bbox, caption) in enumerate(zip(bboxes, captions)):
        target = {
            "boxes": bbox,
            "labels": caption,
        }
        targets.append(target)
    return targets

if __name__ == "__main__":
    
    # Créer les datasets
    train_dataset = CocoDataset(root_dir="../data/coco91/train2017", 
                                    annotation_file="../data/coco91/annotations/captions_train2017.json", 
                                 instance_file="../data/coco91/annotations/instances_train2017.json",
                                    transform=transform)
    val_dataset = CocoDataset(root_dir="../data/coco91/val2017", annotation_file="../data/coco91/annotations/captions_val2017.json", 
                                   instance_file="../data/coco91/annotations/instances_val2017.json",
                                   transform=transform)

    
    batch_size=4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate)
    
    # Initialiser le tokenizer BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Obtenir le token de padding et son ID
    #PAD_TOKEN = tokenizer.pad_token
    PAD_TOKEN = tokenizer.pad_token_id

    # Obtenir le token de début de séquence et son ID
    # Pour BERT, le token de début de séquence est souvent le même que le token [CLS]
    #start_of_sequence_token = tokenizer.cls_token
    PAD_SOS = tokenizer.cls_token_id

    # Obtenir la taille du vocabulaire
    vocab_size = tokenizer.vocab_size

    print(f"Pad token: {PAD_TOKEN}")
    print(f"Start of Sequence token: {PAD_SOS}, ID: {PAD_SOS}")
    print(f"Vocab size: {vocab_size}")
    
    matcher = HungarianMatcher()

    weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}

    losses = ['labels', 'boxes', 'cardinality']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    model = LLMEyaCapModel(num_queries=NUM_QUERIES,vocab_size=vocab_size)
    model = model.to(device)

    criterion = SetCriterion(vocab_size, matcher=matcher, weight_dict=weight_dict, eos_coef = NULL_CLASS_COEF, losses=losses)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_loss = 10**5

    LR = 2e-6
    #LR = 2e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR) #, weight_decay=0.0001)
    EPOCHS=1
    num_queries=NUM_QUERIES
    batch_size=4

    for epoch in range(EPOCHS):
        time_start = time.time()
        train_loss = train_fn(train_loader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
        valid_loss = eval_fn(val_loader, model,criterion, device)

        elapsed = time.time() - time_start
        chk_name = f'LLMEyeCap_01_e{epoch}.bin'
        torch.save(model.state_dict(), chk_name)
        print(f"[Epoch {epoch+1:2d} / {EPOCHS:2d}] Train loss: {train_loss.avg:.3f}. Val loss: {valid_loss.avg:.3f} --> {chk_name}  [{elapsed/60:.0f} mins]")   

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print(f'Best model found in epoch {epoch+1}........Saving Model')
            torch.save(model.state_dict(), 'LLMEyeCap_01_model.bin')
