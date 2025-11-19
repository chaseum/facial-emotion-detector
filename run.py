import argparse
import sys
from src.infer_video import main as vid_main

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    t = sp.add_parser("train")
    t.add_argument("--model", choices=["cnn","backbone"], default="cnn")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--data", default="data")

    sp.add_parser("eval")
    sp.add_parser("cam")
    sp.add_parser("video")

    args, unknown = p.parse_known_args()
    
    if args.cmd == "train":
        from src.train import main as train_main
        train_main([
            f"--model={args.model}",
            f"--epochs={args.epochs}",
            f"--data={args.data}",
        ])

    elif args.cmd == "eval":
        from src.eval import main as eval_main
        eval_main(unknown) 

    elif args.cmd == "cam":
        from src.infer_cam import main as cam_main
        cam_main() 

    elif args.cmd == "video":
        from src.infer_video import main as vid_main
        vid_main(unknown)
        
if __name__ == "__main__":
    main()
