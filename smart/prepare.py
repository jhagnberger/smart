import hydra
from omegaconf import DictConfig
from data.datasets import prepare_dataset


@hydra.main(version_base="1.2", config_path="config", config_name="car")
def main(cfg: DictConfig):
    # Extract config
    config = cfg.experiment

    # Prepare data
    prepare_dataset(config)
    

if __name__ == "__main__":
    main()
    print("Data preparation done. Let's train the model! :)")
