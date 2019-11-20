from model import Model

m = Model(
    "./dataset/train_in.txt",
    "./dataset/train_out.txt",
    "./dataset/test_in.txt",
    "./dataset/test_out.txt",
    "./dataset/vocabs",
    num_units=1024,
    layers=4,
    dropout=0.2,
    batch_size=32,
    learning_rate=0.001,
    output_dir="./dataset/output_couplet",
    restore_model=False,
)

m.train(5000000)
