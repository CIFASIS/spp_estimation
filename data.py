import numpy as np
import theano

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return floatX(Xoh)


def get_batch_indexes(iter, batch_size, max_size):
    n1 = (iter * batch_size) % max_size
    n2 = ((iter + 1) * batch_size - 1) % max_size + 1
    if n1 > n2:
        return np.concatenate((np.arange(n1,max_size),np.arange(0,n2)))
    else:
        return np.arange(n1,n2)

def scale_data(X, max, min, new_min=-1.0, new_max=1.0):
    scale = float(max - min)
    new_scale = float(new_max - new_min)
    return floatX((X - min) * new_scale / scale + new_min)

class Dataset(object):

    def __init__(self, X_train=None, y_train=None,
                 X_valid=None, y_valid=None,
                 X_test =None, y_test=None,
                 X_unlab=None, x_min=0, x_max=255, ymin=0.00, ymax=1.00, nclasses=None, **kw):

        self.__dict__.update(kw)
        del kw # We don't want this in attrs
        self.__dict__.update(locals())
        del self.self # We don't need this either
        if X_train is not None:
            self._nchannels = X_train.shape[1]
            self.first_class = int(y_train.min()) # para corregir si el numero de clase arranca de 1 o 0
            if nclasses is None:
                nclasses = int(y_train.max()) - self.first_class + 1
        elif X_unlab is not None:
            self._nchannels = X_unlab.shape[1]
        else:
            self._nchannels = None


    def get_train_batch(self, index, batch_size, one_hot=True):
        X = self._get_batch(self.X_train,index,batch_size)
        y = self._get_batch(self.y_train,index,batch_size)

        if one_hot:
            y = floatX(OneHot(y.astype(int)-self.first_class,n=self.nclasses))
            return X, self.smooth_labels(y,self.ymin,self.ymax)
        else:
            return X, y.astype(int)-self.first_class


    def get_valid_batch(self, index, batch_size, one_hot=True):
        X = self._get_batch(self.X_valid,index,batch_size)
        y = self._get_batch(self.y_valid,index,batch_size)
        
        if one_hot:
            y = floatX(OneHot(y.astype(int)-self.first_class,n=self.nclasses))
            return X, self.smooth_labels(y,self.ymin,self.ymax)
        else:
            return X, y.astype(int)-self.first_class

    def get_test_batch(self, index, batch_size, one_hot=True):
        X = self._get_batch(self.X_test,index,batch_size)
        y = self._get_batch(self.y_test,index,batch_size)
        
        if one_hot:
            y = floatX(OneHot(y.astype(int)-self.first_class,n=self.nclasses))
            return X, self.smooth_labels(y,self.ymin,self.ymax)
        else:
            return X, y.astype(int)-self.first_class

    def get_unlab_batch(self, index, batch_size):
        X = self._get_batch(self.X_unlab,index,batch_size)
        return X

    def _get_batch(self, X, index, batch_size):
        size = X.shape[0]
        n1 = (index*batch_size)%size
        n2 = ((index+1)*batch_size-1)%size+1
        if n1>n2:
            return floatX(np.concatenate((X[n1:], X[:n2])))
        else:
            return floatX(X[n1:n2])

    def scale_data(self, X, new_min=-1.0, new_max=1.0):
        self.new_min = new_min
        self.new_max = new_max
        scale = self.x_max - self.x_min
        new_scale = new_max - new_min
        return floatX((X-self.x_min)*new_scale/scale+new_min)

    def smooth_labels(self, y, ymin=0.1, ymax=0.9):
        return y*(ymax-ymin)+ymin

    def image_crop(self, X, ph, pw=None, random_state=None):

        if pw is None:
            pw = ph

        h, w = X.shape[2:4]

        if h == ph and w == pw:
            return X

        if random_state:
            j = random_state.random_integers(0, h - ph)
            i = random_state.random_integers(0, w - pw)
        else:
            j = int(round((h - ph)/2.))
            i = int(round((w - pw)/2.))

        return X[:,:,j:j+ph, i:i+pw]

    def inv_scale_data(self, X, old_min=-1.0, old_max=1.0):
        scale = self.x_max - self.x_min
        old_scale = old_max - old_min
        return floatX((X-old_min)*scale/old_scale+self.x_min)

