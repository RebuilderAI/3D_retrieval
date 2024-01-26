import glob

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from data.datasets import *

from utils.tokenizer import SimpleTokenizer
from utils.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from utils.params import parse_args
from utils.logger import setup_logging

from datetime import datetime

import open_clip
import models.uni3d as models

def compute_embedding(clip_model, texts, image):
    text_embed_all = []
    for i in range(texts.shape[0]):
        text_for_one_sample = texts[i]
        text_embed = clip_model.encode_text(text_for_one_sample)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.mean(dim=0)
        text_embed_all.append(text_embed)

    texts = torch.stack(text_embed_all)
    image = clip_model.encode_image(image)
    image = image / image.norm(dim=-1, keepdim=True)
    texts = texts.clone().detach()
    image = image.clone().detach()
    return texts, image

def load_model():
    args = '--model create_uni3d --batch-size 32 --npoints 10000 --num-group 512 --group-size 64 --pc-encoder-dim 512 --clip-model EVA02-E-14-plus  --pretrained laion2b_s9b_b144k --pc-model eva_giant_patch14_560.m30m_ft_in22k_in1k --pc-feat-dim 1408 --embed-dim 1024 --validate_dataset_name modelnet40_openshape --validate_dataset_name_lvis objaverse_lvis_openshape --validate_dataset_name_scanobjnn scanobjnn_openshape --evaluate_3d --ckpt_path ./model.pt'.split()
    
    args, ds_init = parse_args(args)
    global best_acc1

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True 
   
    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])
    
    if ds_init is not None:
        dsconfg_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfg_path, exist_ok=True)
        create_deepspeed_config(args)

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            logging.error("Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    
    # fully initialize distributed device environment
    device = init_distributed_device(args)
    logging.info(f'Running with a single process. Device {args.device}.')

    logging.info("=> create clip teacher...")
    # It is recommended to download clip model in advance and then load from the local
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained, cache_dir = '/data/ansh941/.cache/') 
    clip_model.eval()
    clip_model.to(device)
    
    tokenizer = SimpleTokenizer()

    return clip_model, preprocess, tokenizer
    
clip_model, preprocess, tokenizer = load_model()

def retrieval(image, text):
    global clip_model, preprocess, tokenizer

    image = preprocess(image.convert('L')).unsqueeze(0).cuda()
    
    texts = tokenizer([text])
    if len(texts.shape) < 2:
        texts = texts[None, ...]
    texts = texts.unsqueeze(0).cuda()
    
    image_features, text_features = compute_embedding(clip_model, texts, image)
    features = image_features + text_features
    
    file_list = glob.glob('preprocessed/*.pt')
    logits = []
    for idx, file_path in enumerate(tqdm(file_list)):
        pc_features = torch.load(file_path).cuda()
        logit_per_pc = pc_features.float() @ features.float().t()
        
        logits.append(logit_per_pc.item())
    
    top = np.argsort(logits)[::-1]
    
    return list(map(lambda x: os.path.join('glbs', os.path.basename(x).split('.')[0] + '.glb'), np.asarray(file_list)[top[:5]]))
    
