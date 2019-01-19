import tensorflow as tf;

def add(inputs, name= None):  #Recommend using add_n
    return tf.add(x=inputs[0], y=inputs[1], name= name)

def subtract(inputs, name= None):
    return tf.subtract(x=inputs[0], y=inputs[1], name= name)

def multiply(inputs, name= None):
    return tf.multiply(x=inputs[0], y=inputs[1], name= name)

def divide(inputs, name= None):
    return tf.divide(x=inputs[0], y=inputs[1], name= name)


def matmul(inputs, name= None, **kwargs): #This function is not used in HNet.
    return tf.matmul(inputs[0], inputs[1], name= name, **kwargs)