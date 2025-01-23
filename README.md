IA project

1. download this dataset if you dont have it
    -[text](https://www.physionet.org/content/vindr-spinexr/1.0.0/#files-panel)

2. when the dataset is avaliable, put in in the corresponding folder. Note that some folders might not be in your project(.cache, checkpoint...)
notebooks/
│
├── .cache/
│   └── temp/
│   └── preprocessed_dataset.parquet
│
├── checkpoint/
├── dataset/   <===== HERE!
│
├── utils/
│   ├── training/
│   │   ├── __init__.py
│   │   ├── autoencoder_trainer.py
│   │   ├── checkpoint.py
│   │   ├── dataloader.py
│   │   ├── early_stopping.py
│   │
│   ├── __init__.py
│   ├── create_parquet.py
│   ├── image_grayscaler.py
│   ├── image_normalization.py
│   ├── image_resizer.py
│   ├── load_image.py
│   ├── process_data.py
│   ├── visualize.py
│
├── notebook.ipynb
├── .gitignore
├── pyproject.toml
├── README.md


INSTALATION
dependencies:
$> pip install .
Once the project is installed just run the mind command in the terminal like this:
$> back

License
This project is licensed under the AGPLv3+.