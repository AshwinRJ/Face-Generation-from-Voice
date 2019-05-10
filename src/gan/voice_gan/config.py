import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # misc arguments
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--load_joint_embeddings', default=True,
                        help="Use pretrained joint embeddings")
    parser.add_argument('--load', default='', help="path to saved models")
    parser.add_argument('--save_name', default='', help="Tag name for saved models")
    parser.add_argument('--outf', default='saved_models/',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--joint_path', type=str, default='saved_models/joint_model.pt',
                        help='Path to the pretrained Joint Embedding Model')

    # Training parameters
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3, help='num of channels')

    parser.add_argument('--bs', type=int, default=128, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=15, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--imsize', type=int, default=64,
                        help='the height / width of the input image to network')

    args = parser.parse_args()
    return args
