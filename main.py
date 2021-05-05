from practice_loader import amazon_polarity_data
import time
import yaml
import torch
from pathlib import Path
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.basicClassifier import BasicTextClassifier
from trainer import TextTrainer
from preprocessing import collate_fn, get_vocab


def create_yaml_arg_file(path):
    # args = {"lr": [0.1, 0.01, 0.001], "lr_step": 20, "batch_size": 32,
    #         "num_epochs": 50, "embedding_size": 64}
    args = {"num_epochs": 10, "batch_size": 64, "lr": 2, "lr_step_size": 0.1,
            "data_split": 0.99, "num_classes": 2, "embedding_size": 64, "num_workers": 2}
    # Shuffle does not work with this dataset
    with open(path, 'w') as f:
        doc = yaml.dump(args, f)
        # args = yaml.full_load(f)


def load_data():
    start = time.time()
    data = amazon_polarity_data()
    print("Data load time: {:.2f} s".format(time.time() - start))
    train_data = data["train"]
    print(len(train_data))
    # for j in range(1900000):
    #     next(train_data)
    # for i in range(5):
    #     print(next(train_data))
    return data


def run_basic_classifier(arg_path, model_name, device):
    with open(arg_path, "r") as f:
        args = yaml.full_load(f)
        print(args)
        data = load_data()
        train_size = int(len(data["train"]) * args["data_split"])
        train_data, eval_data = random_split(data["train"], [train_size, len(data["train"]) - train_size])
        train_loader = DataLoader(train_data, batch_size=args["batch_size"], num_workers=args["num_workers"],
                                  collate_fn=collate_fn)
        valid_loader = DataLoader(eval_data, batch_size=args["batch_size"], num_workers=args["num_workers"],
                                  collate_fn=collate_fn)
        test_loader = DataLoader(data["test"], batch_size=args["batch_size"], num_workers=args["num_workers"],
                                 collate_fn=collate_fn)

        vocab, _ = get_vocab(data["train"])
        vocab_size = len(vocab)
        classifier = BasicTextClassifier(vocab_size, args["embedding_size"], args["num_classes"])

        optimizer = optim.SGD(classifier.parameters(), lr=args["lr"])
        scheduler = StepLR(optimizer, gamma=args["lr_step_size"])
        loss_func = torch.nn.BCELoss()
        trainer = TextTrainer(classifier, optimizer, loss_func, device)

        total_time = time.time()
        best_eval = 0.0
        for epoch in range(args["num_epochs"]):
            print("Epoch: ", epoch + 1)
            loss = trainer.train(train_loader)
            cur_eval = trainer.test(valid_loader)
            if cur_eval > best_eval:
                best_eval = cur_eval
                if epoch >= (args["num_epochs"] / 5):
                    torch.save({"state_dict": classifier.state_dict(),
                                "loss": loss, "epoch": epoch + 1},
                               Path.cwd() / "saved_models" / "basic_classifier" / f"{model_name}_{epoch}.pth")
            elif epoch >= (args["num_epochs"] / 5):
                scheduler.step()
        print(f"Total training time: {(time.time()- total_time):.2f}")
        print("Testing metrics")
        trainer.test(test_loader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amazon_path = "args/amazon_sentiment_args.yaml"
    # print(classifier)
    create_yaml_arg_file(amazon_path)
    run_basic_classifier(amazon_path, "amazon_sentiment_classifier", device)
