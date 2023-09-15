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


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Parameters:
    def __init__(self):
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.batch_size = 2
        self.weight_decay = 1e-4
        self.epochs = 300
        self.lr_drop = 200
        self.clip_max_norm = 0.1

args = Parameters()

args.lr=1e-4
args.lr_backbone=1e-5
args.batch_size=32
args.weight_decay=1e-4
args.epochs=300
args.lr_drop=200
args.clip_max_norm=0.1 # type=float,    help='gradient clipping max norm')

# Model parameters
args.frozen_weights=False # ', type=str, default=None, #    help="Path to the pretrained model. If set, only the mask head will be trained")

# * Backbone
args.backbone='resnet50' # type=str, #     help="Name of the convolutional backbone to use")
args.dilation=False  # ', action='store_true',          #      help="If true, we replace stride with dilation in the last convolutional block (DC5)")
args.position_embedding='sine' # type=str, choices=('sine', 'learned'),     help="Type of positional embedding to use on top of the image features")

# * Transformer
args.enc_layers=6 # type=int,      help="Number of encoding layers in the transformer")
args.dec_layers=6 # type=int,       help="Number of decoding layers in the transformer")
args.dim_feedforward=2048  # ===> type=int,   help="Intermediate size of the feedforward layers in the transformer blocks")
args.hidden_dim=256  # ===> type=int,   help="Size of the embeddings (dimension of the transformer)")
args.dropout=0.1   #type=float,   help="Dropout applied in the transformer")
args.nheads=8   #type=int,   help="Number of attention heads inside the transformer's attentions")
args.num_queries=40  #type=int,   help="Number of query slots")
args.pre_norm=True # ', action='store_true')

# * Segmentation
args.masks=False #, action='store_true',  help="Train segmentation head if the flag is provided")

class YANOC(nn.Module): # Im Novel Object Captioning V 0.1 
    
    def __init__(self, backbone, transformer, num_queries, vocab_size):
        
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        
        self.caption_embed = nn.Linear(self.hidden_dim, vocab_size)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone
        '''
        self.capdecoder =  CaptioningDecoder(detr_decoder_dim=transformer.d_model, token_embedding_dim=transformer.d_model, 
                                             vocab_size=vocab_size, num_queries=num_queries, num_layers=6)
        '''
        self.capdecoder = CaptionDecoder(feature_size, token_size, vocab_size,num_queries ).to(device)
        

    def forward(self, samples: NestedTensor, captions):
                            
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)  #featers + position embedding 
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]    
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        outputs_captions=self.capdecoder(hs,captions)
        # predicted_sequences = torch.argmax(outputs_captions, dim=-1)

        out = {'pred_logits': outputs_captions , 'pred_boxes': outputs_coord[-1]}        
        return out
    
    def generate_caption(self, image_path, tokenizer, max_length):
        
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        image = transform(image).unsqueeze(0).to(device)
        #image/=255.
        '''
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #original_shape = img.shape
        #img_input = resize_image(img, target_sizes=params['img_size'])
        
        #out = img_input  / 255.
        
        #image = torch.tensor(np.expand_dims(out,0),dtype=torch.float32)
        #image = image.permute(0,3,1,2).to(device)
        #print(image.shape)
        
        image_pil = Image.fromarray(img)
        # Appliquer les transformations
        image_transformed = transform(image_pil)
        image=image_transformed.unsqueeze(0)
        image=image.to(device)
        '''
        if isinstance(image, (list, torch.Tensor)):
            image = nested_tensor_from_tensor_list(image)
        
        with torch.no_grad():
            features, pos = self.backbone(image)  #featers + position embedding 
            src, mask = features[-1].decompose()
            assert mask is not None
            
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]    
            outputs_coord = self.bbox_embed(hs).sigmoid()
            
            input_ids = torch.ones((1, 40, 1), dtype=torch.long, device=device)
            input_ids.fill_(PAD_SOS)

            
            for i in range(max_length):
                outputs_captions = self.capdecoder(hs, input_ids)
                predicted_sequences = torch.argmax(outputs_captions, dim=-1)
                next_token = predicted_sequences[:, :, -1:]  # take the last token from the sequence
                input_ids = torch.cat((input_ids, next_token), dim=-1)

            #caption = tokenizer.detokenize(input_ids[0].tolist()) #, skip_special_tokens=True)

        return outputs_coord[-1], input_ids # caption[-1]
    

class YANOCModel(nn.Module):
    def __init__(self, num_queries,vocab_size):
        super(YANOCModel,self).__init__()
        self.num_queries = num_queries
        self.vocab_size=vocab_size
        self.backbone = build_backbone(args)
        self.transformer = build_transformer(args)

        self.model = YANOC(
        self.backbone,
        self.transformer,
        num_queries=self.num_queries,
        vocab_size=self.vocab_size
        ) 
        
        # self.in_features = self.caption_embed.in_features   
        
        # self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        
        self.model.num_queries = self.num_queries
        
    def forward(self,images,captions):
        return self.model(images,captions)
    
    def generate_caption(self, image_path, tokenizer, max_length=20):
        return self.model.generate_caption(image_path, tokenizer, max_length)
        

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

def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()
    summary_loss = AverageMeter()    
    
    i = 0
    tk0 = tqdm(range(len(data_loader)))
    
    for i in tk0:
        images, bboxes, captions = data_loader.__getitem__(i)
    
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
    def __init__(self, vocab_size, matcher, weight_dict, eos_coef, losses):
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
        empty_weight = torch.ones(self.vocab_size)
        # empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

        
    '''
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        
        """Classification loss (NLL) for sequences
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes, seq_length]
        """
        #assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # print("src_logits.shape",src_logits.shape)

        # Get the indices for the permutation
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        
        #assert len(src_idx) == len(set(src_idx)), "There are repeated indices in src_idx!"
       

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # print("target_classes_o.shape",target_classes_o.shape)

        # Reshape target tensor to be [batch_size * num_queries, seq_length]
        target_classes_o = target_classes_o.view(-1, target_classes_o.shape[-1])
        
        # print("après reshape target_classes_o.shape",target_classes_o.shape)

        target_classes = torch.full(target_classes_o.shape, PAD_TOKEN,
                            dtype=torch.int64, device=src_logits.device)
        
        #assert max(src_idx) < len(target_classes), "Index out of range in src_idx!"
        
        # print("target_classes.shape",target_classes.shape)
        # assert (target_classes_o < self.vocab_size).all(), "Invalid token index in target_classes_o before filling!"
        # print(f"Vocab size: {self.vocab_size}")

        # print(f"Initial target_classes: {target_classes}")


        for i in range(len(src_idx)):
            # print(f"i: {i}, src_idx[i]: {src_idx[i]}")
            # print(f"target_classes_o[i]: {target_classes_o[i]}")
            target_classes[src_idx[i]] = target_classes_o[i]
            # print(f"Updated target_classes after {i}-th iteration: {target_classes[src_idx[i]]}")
            # print(f"target_classes max value: {target_classes.max()}")
            # print(f"target_classes min value: {target_classes.min()}")


            #assert (target_classes < self.vocab_size).all(), f"Invalid token index detected after {i}-th iteration!"


            
        #print("après le for target_classes.shape",target_classes.shape)

        # Reshape src_logits and target_classes for the loss computation
        # Reshape src_logits to [batch_size * num_queries, seq_length, vocab_size]
        src_logits = src_logits.view(-1, src_logits.shape[2], src_logits.shape[3])


        # Now reshape for the loss
        src_logits = src_logits.reshape(-1, src_logits.shape[-1])
         
        # print("après reshgape 2 src_logits.shape",src_logits.shape)

        target_classes_for_loss = target_classes.view(-1) # This will convert [batch_size, seq_length] to [batch_size*seq_length]
        
        # invalid_indices = target_classes_for_loss[(target_classes_for_loss < 0) | (target_classes_for_loss >= self.vocab_size)]
        # print(f"Invalid indices: {invalid_indices.unique()}")

        
        # assert (target_classes_for_loss >= 0).all() and (target_classes_for_loss < self.vocab_size).all(), "Invalid token index in target!"


        # Ensure the logits and targets are compatible for the loss
        
        # print("Shape of src_logits:", src_logits.shape)
        # print("Shape of target_classes_for_loss:", target_classes_for_loss.shape)

        # assert src_logits.shape[0] == target_classes_for_loss.shape[0], "Logits and targets batch size don't match!"

        loss_ce = F.cross_entropy(src_logits, target_classes_for_loss, ignore_index=PAD_TOKEN)
        losses = {'loss_ce': loss_ce}

        return losses
        '''
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

def eval_fn(data_loader, model,criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()
    
    with torch.no_grad():
        
        i = 0
        tk0 = tqdm(range(len(data_loader)))
    
        for i in tk0:
            images, bboxes, captions = data_loader.__getitem__(i)
        
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

class CaptionDecoder(nn.Module):
    def __init__(self, detr_decoder_dim, token_embedding_dim, vocab_size, num_queries, num_layers=6):
        super(CaptionDecoder, self).__init__()
        
        self.detr_decoder_dim = detr_decoder_dim
        self.token_embedding_dim = token_embedding_dim
        self.vocab_size = vocab_size
        self.num_queries = num_queries

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, token_embedding_dim)
        
        # Initialize GPT-2
        config = GPT2Config(vocab_size=vocab_size, n_embd=detr_decoder_dim + token_embedding_dim, n_head=8 )
        self.gpt2 = GPT2LMHeadModel(config)
        
        self.target_projection = nn.Linear(token_embedding_dim, detr_decoder_dim + token_embedding_dim)
        
    def forward(self, detr_output, captions):
        
        
        # Create an attention mask with shape [batch_size, num_queries, sequence_length]
        attention_mask = (captions != PAD_TOKEN).float().to(captions.device)  # [batch_size, num_queries, sequence_length]


        seq_length = captions.size(2)
        pos_encoding = get_sinusoid_encoding_table(seq_length, self.token_embedding_dim).to(captions.device)
        pos_encoding = pos_encoding.unsqueeze(0).repeat(captions.size(0) * self.num_queries, 1, 1)
        
        # Get the last layer's output from the DETR decoder
        spatial_embedding = detr_output[-1]  # [batch_size, num_queries, detr_decoder_dim]
        
        # Get token embeddings
        token_embeddings = self.token_embedding(captions)  # [batch_size, num_queries, seq_length, token_embedding_dim]
        
        # Repeat the spatial embedding for each token in the sequence and concatenate
        spatial_embedding = spatial_embedding.unsqueeze(2)  # Add seq_length dimension: [batch_size, num_queries, 1, detr_decoder_dim]
        combined_embedding = torch.cat([spatial_embedding.repeat(1, 1, token_embeddings.size(2), 1), token_embeddings], dim=-1)
        # combined_embedding shape: [batch_size, num_queries, seq_length, detr_decoder_dim + token_embedding_dim]
        
        # Prepare the memory for the transformer decoder
        memory = combined_embedding.permute(2, 0, 1, 3).reshape(captions.size(2), -1, self.detr_decoder_dim + self.token_embedding_dim)
        # memory shape: [seq_length, batch_size*num_queries, detr_decoder_dim + token_embedding_dim]
        
        # Prepare the target for the transformer decoder (using token embeddings)
        target = token_embeddings.permute(2, 0, 1, 3).reshape(captions.size(2), -1, self.token_embedding_dim)
        # target shape: [seq_length, batch_size*num_queries, token_embedding_dim]
        
        
        pos_encoding = pos_encoding.permute(1, 0, 2)
        target += pos_encoding


        # Project target to the required dimension
        
        target = self.target_projection(target)
        
        attention_mask = attention_mask.permute(2, 0, 1).reshape(captions.size(2), -1)
        tgt_key_padding_mask = (attention_mask == 0).permute(1,0)
        
        # Prepare the inputs for GPT-2
        inputs_embeds = combined_embedding.permute(2, 0, 1, 3).reshape(captions.size(2), -1, self.detr_decoder_dim + self.token_embedding_dim)
    
        # Reshape attention_mask for GPT-2. Flatten the batch_size and num_queries dimensions.
        attention_mask = attention_mask.reshape(-1, captions.size(2))  # New shape: [batch_size * num_queries, sequence_length]
    
        # Pass through GPT-2
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
    
        # Reshape logits to match the original shape
        logits = logits.view(captions.size(2), captions.size(0), self.num_queries, self.vocab_size).permute(1, 2, 0, 3)
    
        return logits

del model 

model = YANOCModel(num_queries=NUM_QUERIES,vocab_size=vocab_size)
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
batch_size=16

for epoch in range(EPOCHS):
    time_start = time.time()
    train_loss = train_fn(train_loader, model,criterion, optimizer,device,scheduler=None,epoch=epoch)
    valid_loss = eval_fn(val_loader, model,criterion, device)

    elapsed = time.time() - time_start
    chk_name = f'ZeNoCap_04_e{epoch}.bin'
    torch.save(model.state_dict(), chk_name)
    print(f"[Epoch {epoch+1:2d} / {EPOCHS:2d}] Train loss: {train_loss.avg:.3f}. Val loss: {valid_loss.avg:.3f} --> {chk_name}  [{elapsed/60:.0f} mins]")   

    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        print(f'Best model found in epoch {epoch+1}........Saving Model')
        torch.save(model.state_dict(), 'ztod_04_model.bin')

torch.save(model.state_dict(), 'test25.ime')

state_dict = torch.load("ZeNoCap_04_e3.bin")

model.load_state_dict(state_dict)
# model.to(device)

for im in image_paths:
    bb,cc= model.generate_caption( im, tokenizer, max_length=20)
    display_image_ds(im, bb.to('cpu'), cc.to('cpu'))