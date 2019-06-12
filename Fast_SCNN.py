import numpy as np
import tensorflow as tf

def down_sample(input_layer):
    
    ds_layer = tf.keras.layers.Conv2D(32, (3,3), padding='same', strides = (2,2))(input_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    ds_layer = tf.keras.layers.SeparableConv2D(48, (3,3), padding='same', strides = (2,2))(ds_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    ds_layer = tf.keras.layers.SeparableConv2D(64, (3,3), padding='same', strides = (2,2))(ds_layer)
    ds_layer = tf.keras.layers.BatchNormalization()(ds_layer)
    ds_layer = tf.keras.activations.relu(ds_layer)
    
    return ds_layer
    

def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    #x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))
    x = tf.keras.layers.Conv2D(tchannel, (1,1), padding='same', strides = (1,1))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #x = #conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)
    
    x = tf.keras.layers.Conv2D(filters, (1,1), padding='same', strides = (1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)


    if r:
        x = tf.keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

        return x

def global_feature_extractor(lds_layer):
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
    print("gfe_layer.shape:", gfe_layer.shape)
    gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,6,8], gfe_layer.shape[1], gfe_layer.shape[2])
    print("gfe_layer.shape:", gfe_layer.shape)
    
    return gfe_layer


def pyramid_pooling_block(input_tensor, bin_sizes, w, h):
    print(w, h)
    concat_list = [input_tensor]
    #w = 16 # 64
    #h = 16 #32

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), 
                                             strides=(w//bin_size, h//bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)
        print("x in paramid.shape", x.shape)

    concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)



def feature_fusion(lds_layer, gfe_layer):
    ff_layer1 = tf.keras.layers.Conv2D(128, (1,1), padding='same', strides = (1,1))(lds_layer)
    ff_layer1 = tf.keras.layers.BatchNormalization()(ff_layer1)
    #ff_layer1 = tf.keras.activations.relu(ff_layer1)
    print("ff_layer1.shape", ff_layer1.shape)
    
    #ss = conv_block(gfe_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)
    #print(ss.shape, ff_layer1.shape)
    

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    print("ff_layer2.shape", ff_layer2.shape)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    
    print("ff_layer2.shape", ff_layer2.shape)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.activations.relu(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)
    
    print("ff_layer2.shape", ff_layer2.shape)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.activations.relu(ff_final)
    
    print("ff_final.shape", ff_final.shape)
    
    return ff_final

def classifier_layer(ff_final, num_classes):
    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', 
                                                 strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)

    classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', 
                                                 strides = (1, 1), name = 'DSConv2_classifier')(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)
    #change 19 to 20
    #classifier = conv_block(classifier, 'conv', 20, (1, 1), strides=(1, 1), padding='same', relu=True)

    classifier = tf.keras.layers.Conv2D(num_classes, (1,1), padding='same', strides = (1,1))(classifier)
    classifier = tf.keras.layers.BatchNormalization()(classifier)
    classifier = tf.keras.activations.relu(classifier)
    print("classifier.shape", classifier.shape)
    
    classifier = tf.keras.layers.Dropout(0.3)(classifier)
    print("classifier before upsampling:", classifier.shape)

    classifier = tf.keras.activations.softmax(classifier)
    
    return classifier

def get_fast_scnn(w, h, num_classes):
    """
    input image: (w, h)
    """
    
    input_layer = tf.keras.layers.Input(shape=(w, h, 3), name = 'input_layer')
    ds_layer = down_sample(input_layer)
    gfe_layer = global_feature_extractor(ds_layer)
    ff_final = feature_fusion(ds_layer, gfe_layer)
    classifier = classifier_layer(ff_final, num_classes)
    
    fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
    fast_scnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return fast_scnn