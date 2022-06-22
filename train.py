from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


from model import YOLOModule

if __name__ == "__main__":
    # module = YOLOModule()
    # for idx, (x, y) in zip(range(3), module.train_dataloader()):
    #     print(x.shape, y.shape)
    #     y_pred = module(x)
    #     print(y_pred.shape, y[y[..., 4] == 1].shape)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="model-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback])
    module = YOLOModule()
    trainer.fit(module)

    # cli = LightningCLI(
    #     YOLOModule,
    #     trainer_defaults={'gpus': 1, 'callbacks': [checkpoint_callback]},
    #     seed_everything_default=1234,
    #     save_config_overwrite=True
    # )
