import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, Activation, Layer, Flatten


# DeepFM model
class SingleBiasLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(SingleBiasLayer, self).__init__(*args, **kwargs)

    def build(self, units):
        # one single bias value add to batch so shape is (1, )
        self.bias = self.add_weight('bias',
                                    shape=(1, ),
                                    initializer='random_normal',
                                    trainable=True)
                                
    def call(self, X):
        """
        Add the same single bias value to each x
            @X: (batch_size, )
        """
        return X + self.bias


class DeepFM(keras.Model):
    def __init__(self, dim, feature_names, feature_sizes, batch_size, deep_dense_out_dim):
        super(DeepFM, self).__init__()
        self.dim = dim
        self.feature_names = feature_names
        self.feature_sizes = feature_sizes
        self.batch_size = batch_size
        self.init_fm_part()
        self.deep_dense_out_dim = deep_dense_out_dim
        self.init_deep_part()

    def init_features_embeds(self):
        self.features_embeds = []
        for feature_name, feature_size in zip(self.feature_names, self.feature_sizes):
            # feature_size+1 for randomly weight representing unseen feature in train
            # unseen value for one certrain feature in test are all encoded as `feature_size`
            embed_lookup = Embedding(feature_size+1, self.dim, name="embed"+feature_name)
            self.features_embeds.append(embed_lookup)
    
    def init_fm_dense(self):
        """
            Using 1-dim value embed lookup to represent one-hot vector dot weight vector
        """
        self.fm_dense = []
        for feature_name, feature_size in zip(self.feature_names, self.feature_sizes):
            # feature_size+1 for randomly embedding representing unseen feature in train
            # unseen value for one certrain feature in test are all encoded as `feature_size`
            dense_weight_lookup = Embedding(feature_size+1, 1, name="dense"+feature_name)
            self.fm_dense.append(dense_weight_lookup)
        self.fm_bias = SingleBiasLayer()

    def init_fm_part(self):
        self.init_features_embeds()
        self.init_fm_dense()

    def init_deep_part(self):
        self.flatten = Flatten()
        self.deep_dense_layers = []
        # first layer's input size is dim*n_fea, last layer's output size is 1
        for i, output_dim in enumerate(self.deep_dense_out_dim):
            layer = Dense(units=output_dim,
                          activation="relu",
                          use_bias=True,
                          name="deep_dense_"+str(i))
            self.deep_dense_layers.append(layer)

    def fm_part(self, X):
        """
            @X: (batch_size, n_fea)
        """
        # 1-order part (dense and bias):
        one_order = []
        for i in range(X.shape[1]):
            feature_w_lookup = self.fm_dense[i]
            feature_batch_w = feature_w_lookup(X[:, i]) # (batch_size, 1)
            one_order.append(feature_batch_w)
        # (n_fea, batch_size, 1) --stack--> (batch_size, n_fea, 1) --sum&squeeze--> (batch_size, )
        one_order = tf.squeeze(tf.reduce_sum(tf.stack(one_order, axis=1), axis=1))

        # 2-order part:
        embeds = []
        for i in range(len(self.feature_names)):
            feature_embed = self.features_embeds[i]
            batch_embeds = feature_embed(X[:, i]) # (batch_size, dim)
            embeds.append(batch_embeds)
        # (n_fea, batch_size, dim) -> (batch_size, n_fea, dim)
        embeds = tf.stack(embeds, axis=1)
        # feature embeddings crossing dot product
        #two_order = tf.zeros(self.batch_size) # don't use this, for reminder may not be of shape (self.batch_size, )
        two_order = tf.zeros(X.shape[0])
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                # (batch_size, dim)
                V_i, V_j = embeds[:, i, :], embeds[:, j, :]
                # perform batch dot operation between V_i and V_j:
                # (batch_size, 1, dim).dot((batch_size, dim, 1)) --squeeze--> (batch_size, )
                batch_dot = tf.squeeze(tf.matmul(tf.expand_dims(V_i, 1), 
                                                 tf.expand_dims(V_j, -1)))
                two_order += batch_dot

        return self.fm_bias(one_order + two_order) # (batch_size, )

    def deep_part(self, X):
        """
            @X: (batch_size, n_fea)
        """
        embeds = []
        for i in range(X.shape[1]):
            feature_embed = self.features_embeds[i]
            batch_embeds = feature_embed(X[:, i])
            embeds.append(batch_embeds)
        # stack to (n_fea, batch_size, dim) then concat(flat) to (batch_size, n_fea*dim)
        layer_out = self.flatten(tf.stack(embeds, axis=1))
        # feed into dense layers
        for i, layer in enumerate(self.deep_dense_layers):
            layer_out = layer(layer_out)

        return tf.squeeze(layer_out) # (batch_size, )

    def call(self, X):
        """
        Forward function for training
            X: (batch_size, n_fea)
        """
        # (batch_size, )
        return tf.sigmoid(self.fm_part(X) + self.deep_part(X))


# training functions
@tf.function
def train_step(X, Y, model):
    with tf.GradientTape() as tape:
        batch_pred = model(X)
        loss = tf.losses.binary_crossentropy(Y, batch_pred)

    batch_loss = (loss / X.shape[0])

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def train_fm_model(model, dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        total_loss, total_acc, steps = 0, 0, 0

        for (batch_index, (X, Y)) in enumerate(dataset):
            batch_loss = train_step(X, Y, model)
            total_loss += batch_loss
            steps += 1

            if steps % 1000 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.\
                    format(epoch + 1,
                           batch_index,
                           batch_loss))

        print('====== Epoch {} Loss {:.4f} ======'.format(epoch + 1, total_loss / steps))

    return model