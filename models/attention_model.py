import tensorflow as tf

class AttentionModel:
    def __init__(self, num_classes, input_shape, num_attention_units=32, num_dense_units=64):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_attention_units = num_attention_units
        self.num_dense_units = num_dense_units
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Attention mechanism
        attention = tf.keras.layers.Dense(self.num_attention_units, activation='tanh')(inputs)
        attention = tf.keras.layers.Dense(1, activation='softmax')(attention)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.RepeatVector(self.input_shape[0])(attention)
        attention = tf.keras.layers.Permute((2, 1))(attention)

        # Apply attention weights to the input sequence
        weighted_sequence = tf.keras.layers.Multiply()([inputs, attention])

        # LSTM layer
        lstm = tf.keras.layers.LSTM(self.num_dense_units, return_sequences=True)(weighted_sequence)

        # Fully connected layers
        flatten = tf.keras.layers.Flatten()(lstm)
        dense = tf.keras.layers.Dense(self.num_dense_units, activation='relu')(flatten)
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense)

        model = tf.keras.Model(inputs=inputs, outputs=output)

        return model

    def compile_model(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, x_train, y_train, batch_size=32, epochs=10, validation_data=None):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print("Loss:", loss)
        print("Accuracy:", accuracy)