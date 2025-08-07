import argparse


def main(args):
    if args.origin_model == '':
        pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Uncertainty Finetuning Script')

    argparser.add_argument(
        '-b', '-build', '--build',
        dest='build',
        action='store_true',
        help='Run in build mode'
    )

    argparser.add_argument('--model_id',
                           type=str,
                           default='Qwen/Qwen3-8B',
                           help='model_id')

    args = argparser.parse_args()   

    main(args)
