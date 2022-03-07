import os
import tensorflow as tf
import pickle
import re
from datetime import datetime

import model as m
from utils import (
    IcdarTrainingSequence,
    IcdarValidationSequence,
    IcdarEvaluationCallback,
)
from config import cfg

MAX_STEPS = 100000
DECAY_RATE = 0.94


def loss_fun(y_true, y_pred):
    return m.loss(y_true, y_pred)


def train():
    training_name = re.sub("[\-\s\:]", "", str(datetime.utcnow()))[:14]
    model = m.model(freeze=not cfg.unfreeze)
    if cfg.load:
        model.load_weights(cfg.checkpoint_path).expect_partial()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        cfg.learning_rate, decay_steps=MAX_STEPS, decay_rate=DECAY_RATE
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fun,
        metrics=[loss_fun],
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoint/ckpt",
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    history = model.fit(
        IcdarTrainingSequence(),
        validation_data=IcdarValidationSequence(),
        use_multiprocessing=False,
        epochs=cfg.epochs,
        callbacks=[model_checkpoint_callback, IcdarEvaluationCallback(training_name)],
    )
    with open(os.path.join(f"training_results_{training_name}", "history", "wb")) as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    train()
