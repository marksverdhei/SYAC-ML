# TODO

from utils import load_config


def predict_on_dataset(config):
    raise NotImplementedError()


def main():
    config = load_config([])
    predict_on_dataset(config)


if __name__ == "__main__":
    main()