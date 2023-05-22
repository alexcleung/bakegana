import numpy as np
from PIL import Image
import yaml

from dataset import create_dataset


def sample_dataset(config):
    """
    Run the dataset function and visualize a few samples.
    """
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # override config - only load 4 images for example
    config["batch_size"] = 4
        
    train_dataset, val_dataset, label_mapping = create_dataset(config)
    
    for hira, kata, label in train_dataset:
        break
    
    hira = hira.numpy()
    kata = kata.numpy()
    
    hiras = np.split(hira, config["batch_size"])
    katas = np.split(kata, config["batch_size"])
    
    for i, (h,k) in enumerate(zip(hiras, katas)):
        h = np.squeeze(h)
        k = np.squeeze(k)
        Image.fromarray(h).convert('RGB').save(f"h_{i}.png")
        Image.fromarray(k).convert('RGB').save(f"k_{i}.png")
        
    print(label
          
if __name__ == "__main__":
      sample_dataset()