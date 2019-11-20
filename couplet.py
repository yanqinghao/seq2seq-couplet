import argparse
from model import Model

# m = Model(
#     "./dataset/train_in.txt",
#     "./dataset/train_out.txt",
#     "./dataset/test_in.txt",
#     "./dataset/test_out.txt",
#     "./dataset/vocabs",
#     num_units=1024,
#     layers=4,
#     dropout=0.2,
#     batch_size=32,
#     learning_rate=0.001,
#     output_dir="./dataset/output_couplet",
#     restore_model=False,
# )

# m.train(5000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_input_file",
        default="./dataset/train_in.txt",
        type=str,
        required=False,
        help="训练输入文件",
    )
    parser.add_argument(
        "--train_target_file",
        default="./dataset/train_out.txt",
        type=str,
        required=False,
        help="训练输出文件",
    )
    parser.add_argument(
        "--test_input_file",
        default="./dataset/test_in.txt",
        type=str,
        required=False,
        help="测试输入文件",
    )
    parser.add_argument(
        "--test_target_file",
        default="./dataset/test_out.txt",
        type=str,
        required=False,
        help="测试输出文件",
    )
    parser.add_argument(
        "--vocab_file",
        default="./dataset/vocabs",
        type=str,
        required=False,
        help="vob文件",
    )
    parser.add_argument(
        "--num_units", default=1024, type=int, required=False, help="嵌入层输出"
    )
    parser.add_argument("--layers", default=4, type=int, required=False, help="层数")
    parser.add_argument(
        "--dropout", default=0.2, type=float, required=False, help="dropout"
    )
    parser.add_argument("--batch_size", default=32, type=int, required=False, help="bs")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, required=False, help="学习率"
    )
    parser.add_argument(
        "--output_dir",
        default="./dataset/output_couplet",
        type=str,
        required=False,
        help="输出文件夹",
    )
    parser.add_argument(
        "--restore_model", default="False", action="store_true", help="使用预训练模型"
    )
    parser.add_argument(
        "--epochs", default=5000000, type=int, required=False, help="训练轮次"
    )
    parser.add_argument("--start", default=0, type=int, required=False, help="开始轮次")
    args = parser.parse_args()
    m = Model(
        args.train_input_file,
        args.train_target_file,
        args.test_input_file,
        args.test_target_file,
        args.vocab_file,
        num_units=args.num_units,
        layers=args.layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        restore_model=args.restore_model,
    )

    m.train(args.epochs, start=args.start)

