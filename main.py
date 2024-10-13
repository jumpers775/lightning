from RL import RL
import argparse

parser = argparse.ArgumentParser(description='"I am lightning" - Lightning McQueen')

parser.add_argument('-t', '--train', type=int, help='training steps', default=0)
parser.add_argument('-o', '--out', type=str, help='model file', default=None)
parser.add_argument('-m', '--model', type=str, help='model file', default=None)
parser.add_argument('-e', '--eval', type=bool, help='eval model', default=False)

args = parser.parse_args()

if args.train == 0 and not args.eval:
    print("Please specify either training steps or evaluation")
    exit()


if args.train > 0:
    RL(train=args.train, model_path=args.model, out=args.out)

if args.eval:
    RL(model_path=args.model)
