import os
import argparse
import torch
from dataset import get_dataset
from decoder import GPT, GPTConfig
from encoder import get_encoder
from trainer import Trainer, TrainerConfig, Eval
from utils import set_seed
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="test", help="experiment name")
    parser.add_argument("--log_dir", default="../logs", help="/path/to/logs/dir")
    parser.add_argument("--dataset", choices=["webui", "webui_mask"], default="webui", const='bbox',nargs='?')
    parser.add_argument("--data_dir", default="../data/", help="/path/to/dataset/dir")
    parser.add_argument("--device", type=int, default=0)

    # test
    parser.add_argument('--debug', action='store_true', help="debug")
    parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument("--evaluate_layout_path", type=str, default=None)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument("--eval_command", choices=["category_generate", "real_image", "reconstruction"], \
        default="category_generate", const='category_generate',nargs='?', help="real_image indicates to save the real images which labels are given to category_generate or reconstruction.")
    # command args
    parser.add_argument('--save_image', action='store_true', help="save the generated image")
    parser.add_argument('--calculate_coverage', action='store_true', help="calculate the coverage rate")
    parser.add_argument('--calculate_harmony', action='store_true', help="calculate harmony")
    parser.add_argument('--save_pkl', action='store_true', 
                        help="save the generated bbox for heuristic metrics (FID, IoU, Align and Overlap)")

    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument("--labda_kl", type=float, default=0.01)
    parser.add_argument("--labda_gc", type=float, default=0)
    # encoder options
    # parser.add_argument('--encode_dim', default=64, type=int)
    parser.add_argument("--encode_backbone", type=str, default='resnet', help="the backbone used to encode the input images")
    parser.add_argument('--encode_embd', default=2048, type=int)
    parser.add_argument('--finetune_vb', action='store_true', help="finetune the visual backbine on training")
    parser.add_argument('--pretrain_vb', action='store_true', help="pretrain the visual backbine before training")

    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")
    parser.add_argument('--val_sample_num', type=int, default=8, help="the validation samples for visualization")

    args = parser.parse_args()
    if not args.evaluate and not args.debug:
        wandb.init(project="desigen-layout", config=args, name=args.exp)

    log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
    if args.evaluate:
        samples_dir = os.path.join(log_dir, "evaluate_samples")
    else:
        samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(os.path.join(samples_dir, args.eval_command), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    cur_data = 'train' if not args.debug else 'val'
    train_dataset = get_dataset(args.dataset, cur_data, args.data_dir)
    if args.evaluate:
        valid_dataset = get_dataset(args.dataset, "test", args.data_dir)
    else:
        valid_dataset = get_dataset(args.dataset, "val", args.data_dir)

    print("train dataset vocab_size: ",train_dataset.vocab_size)
    print("train dataset max_length: ", train_dataset.max_length)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      encode_embd=args.encode_embd,
                      component_class=train_dataset.component_class) 
    decoder_model = GPT(mconf)
    encoder_model = get_encoder(name=args.encode_backbone, grad=(args.finetune_vb is not None), pretrain=args.pretrain_vb)
    tconf = TrainerConfig(dataset=args.dataset,
                          debug=args.debug,
                          max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate=args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every,
                          encoder_path=args.encoder_path,
                          decoder_path=args.decoder_path,
                          device=args.device,
                          val_sample_num=args.val_sample_num,
                          labda_kl=args.labda_kl,
                          labda_gc=args.labda_gc,
                          evaluate_layout_path=os.path.join(log_dir, "generated_layout.pth"))

    if args.evaluate:
        print("testing...")
        command = {"name": args.eval_command, "save_image": args.save_image, 
        "calculate_coverage": args.calculate_coverage, "save_pkl": args.save_pkl, 
        "calculate_harmony": args.calculate_harmony}
        tconf.evaluate_layout_path=os.path.join(log_dir, "generated_layout_%s.pth" % args.eval_command)
        evaler = Eval(encoder_model, decoder_model, valid_dataset, tconf)
        evaler.eval(command)
    else:
        print('length of training set: ', len(train_dataset))
        trainer = Trainer(encoder_model, decoder_model, train_dataset, valid_dataset, tconf)
        trainer.train()
