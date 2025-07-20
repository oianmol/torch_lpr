from torch_train_emnist import EMNISTTrainer

if __name__ == "__main__":
    trainer = EMNISTTrainer(batch_size=36)
    trainer.load_and_predict()
