## Articles classification

Navigate to repository folder and install dependencies with:

## Install and run

```
pip install -r requirements.txt        
```

## Code structure

1. train.py
Entry point of the program. Contains training module for the models. Model parameters
can be changed in trainer initialization:


```
trainer = Trainer(
        model="transformer",
        max_epochs=100,
        batch_size=16,
        lr=0.001,
        data_root="data",
        columns=["lemma_title", "lemma_description"],  # lemma_maintext
        input_size=16,
        hidden_size=16)     
```

It is also possible to inspect results with TensorBoard, navigate to repository folder in terminal
and exectue:

```
tensorboard --logdir runs    
```

2. data.py Contains dataset text encoder, dataset, dataloader and datamodule


3. model.py Contains LSTM and Transformer models and positional encoding module