import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, NonlinearityLayer, GlobalPoolLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import FlattenLayer
from lasagne.nonlinearities import softmax, linear
# from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import leaky_rectify as lrelu

def build_representation(img_size=[64,64], nchannels=3, ndf=64,
                     vis_filter_size=5, filters_size=5, global_pool=True, strides = [2, 2, 2, 2]):
    print 'cnn'
    #if img_size[0] % 32 is not 0 or img_size[1]!=img_size[0]:
    #    # La imagen debe ser cuadrada y multiplo de 32
    #    raise 1

    depth = len(strides)
    w_sizes = [filters_size] * depth
    w_sizes[0] = vis_filter_size

    X = InputLayer((None, nchannels, img_size[0], img_size[1]))
    ishape = lasagne.layers.get_output_shape(X)
    # print ishape

    wf = 1
    h = X
    for i, s in enumerate(strides):
        wf *= s
        filter_size = w_sizes[i]
        x1 = Conv2DLayer(h, num_filters=wf * ndf, filter_size=filter_size, stride=s, pad='same',
                         b=None, nonlinearity=None, name='cnn_l%d_Conv'%i)
        x2 = BatchNormLayer(x1, name='cnn_l%d_BN'%i)
        h = NonlinearityLayer(x2, nonlinearity=lrelu)
        ishape = lasagne.layers.get_output_shape(x1)
        # print ishape

    if global_pool:
        h = GlobalPoolLayer(h, pool_function=T.max, name='cnn_last_code')
    else:
        h = FlattenLayer(h, name='cnn_last_code')


    return h

def build_classifier(nclasses, img_size=[64,64], nchannels=3, ndf=64,
                     vis_filter_size=5, filters_size=5, global_pool=True, strides = [2, 2, 2, 2]):

    h = build_representation(img_size, nchannels, ndf, vis_filter_size, filters_size, global_pool, strides)

    y = DenseLayer(h, num_units=nclasses, nonlinearity=softmax, name='softmax')

    return y


def build_classifier_and_regressor(nclasses, img_size=[64, 64], nchannels=3, ndf=64,
                     vis_filter_size=5, filters_size=5, global_pool=True, strides=[2, 2, 2, 2]):
    h = build_representation(img_size, nchannels, ndf, vis_filter_size, filters_size, global_pool, strides)

    c = DenseLayer(h, num_units=nclasses, nonlinearity=softmax, name='softmax')

    r = DenseLayer(h, num_units=1, nonlinearity=linear, name='linear_out')

    return c, r
