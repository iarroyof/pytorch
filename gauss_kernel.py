# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from pdb import set_trace as st

def kernel_product(w, x, mode = "gaussian", s = 0.1):
    w_i = torch.t(w).unsqueeze(1)
    x_j = x.unsqueeze(0)
    xmy = ((w_i - x_j)**2).sum(2)
    #st()
    if   mode == "gaussian" : K = torch.exp( - (torch.t(xmy) ** 2) / (s**2) )
    elif mode == "laplace"  : K = torch.exp( - torch.sqrt(torch.t(xmy) + (s**2)))
    elif mode == "energy"   : K = torch.pow(   torch.t(xmy) + (s**2), -.25 )

    return K


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 300, 100, 2
n_epochs = 10000

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
# Create random scalar for kernel bandwidth
s1 = Variable(torch.randn(1, H).type(dtype), requires_grad=True)
s2 = Variable(torch.randn(1, D_out).type(dtype), requires_grad=True)
b1 = Variable(torch.randn(H, ).type(dtype), requires_grad=True)
b2 = Variable(torch.randn(D_out, ).type(dtype), requires_grad=True)


learning_rate = 1e-6
for t in range(n_epochs):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations on Variables; we compute
    # ReLU using our custom autograd operation.
#    y_pred = relu(x.mm(w1)).mm(w2)
    y_pred = relu(kernel_product(w2, relu(kernel_product(w1, x, "gaussian", s1) + b1), "gaussian", s2) + b2 )
#    y_pred = relu(kernel_product(w1, x, "gaussian", s1) + b1).mm(w2)
    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    s1.data -= learning_rate * s1.grad.data
    s2.data -= learning_rate * s2.grad.data
    b1.data -= learning_rate * b1.grad.data
    b2.data -= learning_rate * b2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    s1.grad.data.zero_()
    s2.grad.data.zero_()
    b1.grad.data.zero_()
    b2.grad.data.zero_()
