import tensorflow as tf
import logging

logging.getLogger()
tf.compat.v1.random.set_random_seed(0)


def get_weights(current_weight):
    # print('Getting weights',self.neuronBlock)
    temp_weights = []
    weight_shapes = []
    for weights in current_weight:
        try:
            w = weights.numpy().astype(np.float)

            temp_weights.extend(list(w.flatten()))

            weight_shapes.append(w.shape)
            # print(list(w.shape))

            # nb.weights[0].assign(w*0)
        except Exception as e:
            print("Getting weights, failed because", e)
    # print('Len weights:',len(temp_weights))

    return temp_weights, weight_shapes


def get_weight_entire(tf_model_array):

    weights_array = []

    for model_ in tf_model_array:

        weights_array.append(get_weights(model_.weights))

    return weights_array


def set_weight_entire(weights_array, tf_model_array):

    for weights_array, model_ in zip(weights_array, tf_model_array):
        set_weights(weights_array, model_.weights)

def shape_length(input_array):
    """ Dumb function to get length of tuple """
    # print(input_array)
    if len(input_array) == 1:
        return input_array[0], len(input_array)
    if len(input_array) == 2:
        return input_array[0] * input_array[1], len(input_array)
    if len(input_array) == 3:
        return input_array[0] * input_array[1] * input_array[2], len(input_array)


def set_weights_v2(weight_array_to_set, weights, shape_array):
    # print('Getting weights',self.neuronBlock)
    


    weight_array_temp = weight_array_to_set
    offset = 0
    for shape, weight in zip(shape_array, weight_array_to_set):
        #try:
        if 1:
            # w = weights.numpy()

            # shape_list = list(w.shape)


           
            length_weight, shape_dim = shape_length(shape)
            
            # print("Lenght WEIGHT:{}".format(length_weight))

            w_set = weights[offset:offset+length_weight]

            w_set_reshape = np.reshape(np.array(w_set), np.array(shape))

            weight_array_temp = weight_array_temp[length_weight:]

            # temp_weights.extend(list(w.flatten()))
            # print(list(w.shape))
            
            #print('w_set_reshape : {}'.format(w_set_reshape.shape))
            
            weight.assign(w_set_reshape)
            offset = length_weight
            
        #except Exception as e:
        #    logging.warning("Set weights failed because", e)
    # print('Len weights:',len(weight_array))

def set_weights(weight_array_to_set, previous_weight):
    # print('Getting weights',self.neuronBlock)
    def multiply(input_array):
        """ Dumb function """
        # print(input_array)
        if len(input_array) == 1:
            return input_array[0], len(input_array)
        if len(input_array) == 2:
            return input_array[0] * input_array[1], len(input_array)
        if len(input_array) == 3:
            return input_array[0] * input_array[1] * input_array[2], len(input_array)


    weight_array_temp = weight_array_to_set
 
    for weights in previous_weight:
        #try:
        if 1:
            w = weights.numpy()

            shape_list = list(w.shape)


           
            length_weight, shape_dim = multiply(shape_list)
            
            # print("Lenght WEIGHT:{}".format(length_weight))

            w_set = weight_array_temp[0:length_weight]

            w_set_reshape = np.reshape(np.array(w_set), np.array(shape_list))

            weight_array_temp = weight_array_temp[length_weight:]

            # temp_weights.extend(list(w.flatten()))
            # print(list(w.shape))
            
            #print('w_set_reshape : {}'.format(w_set_reshape.shape))

            weights.assign(w_set_reshape)
        #except Exception as e:
        #    logging.warning("Set weights failed because", e)
    # print('Len weights:',len(weight_array))


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name="W", shape=(input_shape[-1], self.units), initializer="zero"
        )
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zero")

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class modifierBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(modifierBlock, self).__init__()

        self.linear = Linear(8)

        self.conv1d = tf.keras.layers.Conv1D(1, 1, padding="valid")
        self.conv2d = tf.keras.layers.Conv1D(3, 1, padding="valid")
        self.conv3d = tf.keras.layers.Conv1D(1, 1, padding="valid")

        self.maxpool1 = tf.keras.layers.MaxPool1D()

        self.conv_out = None  # tf.keras.layers.Conv1D(3,3,padding='same')

        self.linear_out = None

        self.output_nn = None

        self.flatten = tf.keras.layers.Flatten()

        self.linear2 = tf.keras.layers.Dense(
           2
        )

        self.linear1 = tf.keras.layers.Dense(
           2
        )

        self.linear_in1 = tf.keras.layers.Dense(
           1
        )


        self.linear_in2 = tf.keras.layers.Dense(
           1
        )


    def call(self, inputs, inputs2):

        # print(inputs, inputs2)

        #inputs = self.linear_in1(inputs)
        #inputs2 = self.linear_in1(inputs2)
        x = tf.keras.layers.concatenate([inputs, inputs2])  # , axis=0)


        #x = inputs2#tf.keras.layers.concatenate([inputs2])  # , axis=0)


        x = tf.expand_dims(x, axis=-1)
        x = self.linear2(x)

        x = tf.nn.tanh(x)

        x = self.linear1(x)

        # x = tf.nn.tanh(x)
        #print('1', x.shape)
        # x = self.conv1d(x)
        

        # x = tf.nn.tanh(x)

        # x = self.maxpool1(x)

        # x = tf.nn.tanh(x)

        # x = self.conv2d(x)

        # x = tf.nn.tanh(x)

        # #print('2', x.shape)

        # x = tf.nn.tanh(x)

        # x = self.conv3d(x)

        # #print('3', x.shape)

        # x = tf.nn.tanh(x)


        #print(x)

        # x = self.conv2d(x)
        # x = tf.nn.tanh(x)

        # x = self.conv3d(x)
        # x = tf.nn.tanh(x)

        # x2 = self.linear

        # x = self.linear_2(x)
        # x = tf.nn.relu(x)
        # x = self.linear_2(x)
        # x = tf.nn.relu(x)
        # x = self.linear_2(x)
        # x = tf.nn.relu(x)
        # x = self.linear_3(x)print('input',inputs.shape.as_list())
        # print('-----------', inputs.shape.as_list())
        if 1:
            #if self.conv_out is None:
            #    self.conv_out = tf.keras.layers.Conv1D(
            #       1,inputs.shape.as_list()[1], 
            #    )  # tf.keras.layers.Dense(inputs.shape.as_list()[1])#tf.keras.layers.Conv1D(inputs.shape.as_list()[1]),3,padding='same')

            # x = self.conv_out(x)
            # print('final', x.shape)

            if self.output_nn is None:
                self.output_nn = tf.keras.layers.Dense(inputs.shape.as_list()[1])#, activation='tanh')

        x = self.flatten(x)
        x = self.output_nn(x)
        x = tf.nn.tanh(x)

        # x = tf.nn.sigmoid(x)

        def f1():
            return tf.constant(0)

        def f2():
            return tf.constant(1)

        return x  # tf.math.reduce_sum(x)#tf.cond(tf.less(tf.math.reduce_sum(x), tf.constant(0.5)), f1, f2 )
        #return tf.cond(tf.less(tf.math.reduce_sum(x), tf.constant(0.9)), f1, f2 )

        # eturn tf.nn.sigmoid(x)


class neuronBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(neuronBlock, self).__init__()

        self.linear = tf.keras.layers.Dense(
            3  , kernel_initializer='RandomNormal', bias_initializer='RandomNormal')#Linear(8)
        self.linear1 = tf.keras.layers.Dense(
            3
        )  # ,
        self.linear2 = tf.keras.layers.Dense(
            5
        )  # , kernel_initializer='RandomNormal', bias_initializer=tf.keras.initializers.constant(1.0))#Linear(8)
        self.linear3 = tf.keras.layers.Dense(
            3
        )  # , kernel_initializer='RandomNormal', bias_initializer=tf.keras.initializers.constant(1.0))#Linear(8)
         #kernel_initializer='RandomNormal', bias_initializer=tf.keras.initializers.constant(1.0))#Linear(1)
        self.out = tf.keras.layers.Dense(1)
    def call(self, inputs):
        x = self.linear(inputs)
        x = tf.nn.tanh(x)
        
        def f1():
            #return tf.math.reduce_sum(x)
            return tf.constant(1.0)

        def f2():
            return tf.constant(-1.0)

        # return tf.math.reduce_sum(
        #     x
        # )  
        #return tf.cond(tf.less(tf.math.reduce_sum(x), tf.constant(0.9)), f1, f2 )
        #x = self.out(x)
        x = tf.math.reduce_sum(tf.nn.tanh(x))
        return x
        #return tf.math.reduce_sum(x) #tf.nn.sigmoid(tf.math.reduce_sum(x))


@tf.function
def add(a, b):
    return a + b


@tf.function
def linear_layer(m, x, b):
    return add(tf.matmul(m, x), b)  # mx + b


@tf.function
def relu(x):
    return tf.nn.relu(x)


@tf.function
def caller(input):
    # nb = neuronBlock()
    x = linear_layer(input)
    return x


import numpy as np
import time

if __name__ == "__main__":

    nb = modifierBlock()  # modifierBlock()
    # print(nb.get_config())
    start = time.time()
    # caller(tf.ones(shape=(1,30)))
    c = tf.zeros(shape=(1, 200))
    print(nb(c, c))  # , tf.ones(shape=(1,30))))

    # c = tf.ones(shape=(1,200))
    # print(nb(c))#, tf.ones(shape=(1,30))))

    # c = tf.ones(shape=(1,200))*0.2
    # print(nb(c))#, tf.ones(shape=(1,30))))

    # c = tf.ones(shape=(1,200))*10000
    # print(nb(c))#, tf.ones(shape=(1,30))))
    print("time", time.time() - start)
    # print(nb.weights)
    for weights in nb.weights:
        try:
            w = weights.numpy()
            print(list(w.shape))
            # print(list(w.shape))

            # @nb.weights[0].assign(w*0)
            weights.assign(w * 0)
        except:
            pass

    for weights in nb.weights:
        try:
            w = weights.numpy()
            # print(w)

            # weights.assign(w*0)
        except:
            pass

    # print(nb(tf.constant([1.038944542, 1.038944542, 0.038944542, 0.038944542, 0.038944542], shape=(1, 5), dtype=tf.float32)))
# result = nb([1,2,3])
# print(result)
