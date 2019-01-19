import tensorflow as tf;

def sparse_softmax_cross_entropy(labels, logits, **kwargs):
    return tf.losses.sparse_softmax_cross_entropy(
        labels = tf.squeeze(labels, axis=-1),
        logits = logits,
        **kwargs
        )