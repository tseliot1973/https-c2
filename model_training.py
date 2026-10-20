from pandas import read_csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from argparse import ArgumentParser
from sys import exit
import datetime
from tools import FEATURE_LIST


def main():
    parser = ArgumentParser(description="Train Model from input CSV")
    parser.add_argument("-i", "--input", required=True, help="Input Training Data")
    parser.add_argument("-m", "--model", required=True, help="Name for output model file")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("-g", "--gamma", type=int, default=1, help="Gamma value for the focal loss")
    parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs for training")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    # define variables for model metadata
    #features = # number of features in input data
    features = len(FEATURE_LIST)
    batch_size = args.batch_size
    epochs = args.epochs
    smaller_layers = 512
    larger_layer = 2048

    if not ((args.model).endswith(".keras")):
        print("Model name must end with .keras")
        exit(1)

    training_data = read_csv(args.input, dtype=np.float32)
    training_labels = training_data.pop('label')
    training_data = training_data[FEATURE_LIST].reset_index(drop=True)
    

    # split training and test data
    data_train, data_test, label_train, label_test = train_test_split(
    training_data, training_labels, test_size=0.2, random_state=42 )

    print(f"training_labels.shape: {training_labels.shape}")
    print(f"training_data.shape: {training_data.shape}")
    print(f"First five rows of training_data:\n{training_data[:5]}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model,
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]


    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(data_train))

    model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(features,)),
     normalizer,
    tf.keras.layers.Dense((smaller_layers), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense((larger_layer), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense((larger_layer), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense((smaller_layers), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # dropped an order of magnitude from previous test
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=float(args.gamma))
    #loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=[
            #'accuracy',
            tf.keras.metrics.AUC(curve='PR', name='pr_auc')
        ]
    )
    model.summary()
    model.fit(data_train, label_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(data_test, label_test),
    callbacks=callbacks)

    print(f"Best checkpoint saved as file {args.model}")

if __name__ == "__main__":
    main()
