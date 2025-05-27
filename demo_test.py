# MFLVC
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)

#GcFAgg
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=100)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)

#DealMVC
Dataname = 'BBCSport'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', type=int, default=544)
parser.add_argument("--temperature_f", type=float, default=0.5)
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--mse_epochs", type=int, default=300)
parser.add_argument("--con_epochs", type=int, default=100)
parser.add_argument("--tune_epochs", type=int, default=50)
parser.add_argument("--feature_dim", type=int, default=512)
parser.add_argument("--high_feature_dim", type=int, default=512)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=544)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 用来初始化随机种子，后期测试不同数据使用
if args.dataset == "MNIST-USPS":
    args.fine_tune_epochs = 100
    seed = 10
if args.dataset == "CCV":
    args.fine_tune_epochs = 100
    seed = 3
if args.dataset == "Hdigit":
    args.fine_tune_epochs =100
    seed = 10
if args.dataset == "YouTubeFace":
    args.fine_tune_epochs = 100
    seed = 10
if args.dataset == "Cifar10":
    args.fine_tune_epochs = 10
    seed = 10
if args.dataset == "Cifar100":
    args.fine_tune_epochs = 200
    seed = 10
if args.dataset == "Prokaryotic":
    args.fine_tune_epochs = 50
    seed = 10
if args.dataset == "Synthetic3d":
    args.fine_tune_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.fine_tune_epochs = 100
    seed = 10
if args.dataset == "Caltech-3V":
    args.fine_tune_epochs = 100
    seed = 10
if args.dataset == "Caltech-4V":
    args.fine_tune_epochs = 150
    seed = 10
if args.dataset == "Caltech-5V":
    args.fine_tune_epochs = 200
    seed = 5



criterion = Loss(args.batch_size, args.temperature_f, device).to(device)

criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)