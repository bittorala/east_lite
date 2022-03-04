import tensorflow as tf
from utils import IcdarTrainingSequence, FLAGS, IcdarValidationSequence, IcdarEvaluationCallback
import pickle

import model as m

MAX_STEPS = 100000
DECAY_RATE = 0.94

def loss_fun(y_true, y_pred):
    return m.loss(y_true, y_pred)


def load_and_train():
    model = m.model(freeze=False)
    model.load_weights(FLAGS.checkpoint_path).expect_partial()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.learning_rate, decay_steps=MAX_STEPS, decay_rate=DECAY_RATE
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fun,
        metrics=[loss_fun],
    )

    history = model.fit(
        IcdarTrainingSequence(),
        validation_data=IcdarValidationSequence(),
        use_multiprocessing=False,
        epochs=FLAGS.epochs,
        callbacks=[IcdarEvaluationCallback(1)],
    )
    with open(f"history_ft", "wb") as f:
        pickle.dump(history.history, f)


def train():
    # strategy = tf.distribute.MirroredStrategy()
    # print(f'Number of devices: {strategy.num_replicas_in_sync}')

    # with strategy.scope():
    model = m.model(freeze=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.learning_rate, decay_steps=MAX_STEPS, decay_rate=DECAY_RATE, staircase=True
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
        epochs=FLAGS.epochs,
        callbacks=[model_checkpoint_callback, IcdarEvaluationCallback()],
    )
    with open(f"history", "wb") as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    if FLAGS.load_and_train:
        load_and_train()
    else:
        train()
