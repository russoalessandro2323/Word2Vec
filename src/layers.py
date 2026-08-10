import numpy as np

class Layer:
  def forward(self, inputs):
      raise NotImplementedError
  def backward(self, output_gradient):
      raise NotImplementedError

class EmbeddingLayer(Layer):
  def __init__(self, vocab_size, embedding_dim):
    self.w = np.random.randn(vocab_size, embedding_dim) * np.sqrt(2.0 / vocab_size) # w_in (vocab_size, embedding_dim)
    self.dw = np.zeros(self.w.shape)
    self.input = None

  def forward(self, inputs):
    self.input = inputs
    return self.w[inputs] # (input_len, embedding_dim)

  def backward(self, output_gradient):
    # output_gradient       (input_len, embedding_dim)
    # dw                    (vocab_size, embedding_dim)
    # input                 (input_len)
    # self.dw[self.input]   (input_len, embedding_dim)
    self.dw = np.zeros_like(self.w)
    for idx, word in enumerate(self.input):
      self.dw[word] += output_gradient[idx]

class DenseLayer(Layer):
  def __init__(self, fan_in, fan_out):
    self.w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_out) # (embedding_dim, vocab_size)
    self.b = np.zeros((fan_out, 1))  # (vocab_size, 1)
    self.dw = None
    self.db = None
    self.input = None

  def forward(self, inputs):
    self.input = inputs # (input_len, embedding_dim)
    z = inputs @ self.w + self.b.T  # (input_len, embedding_dim) @ (embedding_dim, vocab_size) = (input_len, vocab_size)
    return z # (input_len, vocab_size)

  def backward(self, out_grad):
    # out_grad (input_len, vocab_size)
    N = self.input.shape[1] # embedding_dim

    self.db =  np.sum(out_grad, axis=0, keepdims=True).T # (vocab_size, 1)
    self.dw =  self.input.T @ out_grad        # (embedding_dim, vocab_size)

    return out_grad @ self.w.T  # (input_len, embedding_dim)

class SoftMaxCrossEntropy(Layer):
  def __init__(self):
    self.output = None
    self.y_true = None

  def forward(self, inputs, y_true):
    self.y_true = y_true  # (input_len, vocab_size)

    # 1. Softmax su axis=1 (lungo le classi)
    shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
    exps = np.exp(shifted_inputs)
    self.output = exps / np.sum(exps, axis=1, keepdims=True)

    # 2. Cross Entropy Loss corretta
    eps = 1e-15
    clipped_values = np.clip(self.output, eps, 1 - eps)

    # np.sum su axis=1 isola la probabilità della parola corretta per OGNI campione.
    # np.mean fa la media tra tutti i campioni del batch.
    loss_per_sample = -np.sum(self.y_true * np.log(clipped_values), axis=1)
    return np.mean(loss_per_sample)

  def backward(self, out_grad=None):
    # gradiente analitico dZ = (P - Y) / batch_size
    batch_size = self.y_true.shape[0]
    return (self.output - self.y_true) / batch_size