import tensorflow as tf

class GRUModel:
    def __init__(self, num_classes, input_shape, num_units=64):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_units = num_units
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(self.num_units, input_shape=self.input_shape))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

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