import torch
import numpy as np
import scipy.special as fun

torch.set_default_dtype(torch.double)

class Bessel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        return (torch.from_numpy(np.array(fun.kv(nu,inp.detach().numpy()))))

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        grad_in = grad_out.numpy() * np.array(fun.kvp(nu,inp.detach().numpy()))
        return (torch.from_numpy(grad_in), None)

class Matern(object):

    def __init__(self, nu = 2.01, lengthscale=np.log(2)):
        self.nu = nu
        self.log_lengthscale = torch.tensor(np.log(lengthscale), dtype=torch.double)

    def _set_lengthscale(self, lengthscale):
        self.log_lengthscale.requires_grad_(False)
        self.log_lengthscale = torch.tensor(np.log(lengthscale))

    def lengthscale(self):
        self.log_lengthscale.requires_grad_(False)
        return (torch.exp(self.log_lengthscale).item())

    def A(self, nu, r):
        r_ = r.clamp(1e-15)
        A_ = torch.pow(r_,nu) * Bessel.apply(r_,nu)
        return (A_)

    def forward(self, x1, x2 = None, **params):
        self.log_lengthscale.requires_grad_(True)
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None: x2 = x1
        r_ = (x1.reshape(-1,1) - x2.reshape(1,-1)).abs()
        r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
        A_ = self.A(self.nu, r_)
        C_ = np.power(2,1-self.nu) * np.exp(-fun.loggamma(self.nu)) * A_
        return (C_)

    def C(self, x1, x2 = None):
        return (self.forward(x1,x2).detach())

    def dC_dx1(self, x1, x2 = None):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if x2 is None: x2 = x1
        with torch.no_grad():
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            dA_ = -r_ * self.A(self.nu-1, r_)
            dC_ = np.power(2,1-self.nu) * np.exp(-fun.loggamma(self.nu)) * dA_
            dC_ = dC_ * np.sqrt(2*self.nu) / lengthscale
            dC_ = dC_ * torch.sign(dist_)
        return (dC_)

    def dC_dx2(self, x1, x2 = None):
        return (-self.dC_dx1(x1,x2))

    def d2C_dx1dx2(self, x1, x2 = None):
        lengthscale = torch.exp(self.log_lengthscale)
        x1 = x1.squeeze()
        if (x2 is None): x2 = x1
        with torch.no_grad():
            dist_ = (x1.reshape(-1,1) - x2.reshape(1,-1))
            r_ = dist_.abs()
            r_ = np.sqrt(2.*self.nu) * r_ / lengthscale
            dA2_ = -self.A(self.nu-1,r_) + torch.pow(r_,2.0) * self.A(self.nu-2,r_)
            dC2_ = np.power(2,1-self.nu) * np.exp(-fun.loggamma(self.nu)) * dA2_
            dC2_ = dC2_ * (2. * self.nu / torch.square(lengthscale))
            dC2_ = -dC2_
        return (dC2_)

def GPTrain(train_x, train_y, kernel=None, noisy=True, 
            max_iter=500, eps=1e-6, criterion='EB', verbose=False):
    # preprocess input data
    n = train_x.size(0)
    # normalized x to 0 and 1
    x_range = [torch.min(train_x).item(), torch.max(train_x).item()]
    train_x = (train_x - x_range[0]) / (x_range[1] - x_range[0])
    # set up kernel
    if (kernel is None):
        kernel = Matern(nu=2.01, lengthscale=1./(n-1))
    # set up optimizer
    if (noisy):
        # lambda = noise/outputscale
        log_lambda = torch.tensor(np.log(1e-1))
        log_lambda.requires_grad_(True)
        optimizer = torch.optim.Adam([kernel.log_lengthscale,log_lambda], lr=1e-2)
    else:
        # nugget term to avoid numerical issue
        log_lambda = torch.tensor(np.log(1e-6))
        optimizer = torch.optim.Adam([kernel.log_lengthscale], lr=1e-2)
    # training
    prev_loss = np.Inf
    for i in range(max_iter):
        optimizer.zero_grad()
        R = kernel.forward(train_x) + torch.exp(log_lambda) * torch.eye(n)
        e, v = torch.linalg.eigh(R) # eigenvalue and eigenvectors
        # e = e.real.double() # only take the real part, ignore imaginary part
        # v = v.real.double() # only take the real part, ignore imaginary part
        e = e.double()
        v = v.double()
        a = v.T @ torch.ones(n)
        b = v.T @ train_y
        mean = ((a/e) @ b) / ((a/e) @ a)
        d = v.T @ (train_y - mean)
        if (criterion == "EB"):
            # empirical bayes objective
            outputscale = 1./n * (d/e) @ d
            loss = n*torch.log(outputscale) + torch.sum(torch.log(e))
        elif (criterion == 'CV'):
            # loocv objective
            loocv = (1/(v.square() @ (1/e))) @ d
            loss = loocv.square().mean()
        else:
            print("no criterion %s, only supporting EB (empirical Bayes) or CV (LOOCV)" %(criterion))
            return (None)
        # backprop
        loss.backward()
        optimizer.step()
        if (log_lambda < np.log(1e-6)):
            log_lambda = np.log(1e-6)
        # early termination check every 25 iterations
        if ((i+1)%25 == 0):
            if (verbose):
                print('Iter %d/%d - Loss: %.3f' % (i+1, max_iter, loss.item()))
                print('lengthscale: %.2f' %(kernel.lengthscale()))
                print('lambda: %.2f' %(torch.exp(log_lambda).item()))
            if (np.nan_to_num((prev_loss-loss.item())/abs(prev_loss),nan=1) > eps):
                prev_loss = loss.item()
            else:
                if (verbose): print('Early Termination!')
                break
    with torch.no_grad():
        R = kernel.C(train_x) + torch.exp(log_lambda) * torch.eye(n)
        e, v = torch.linalg.eigh(R) # eigenvalue and eigenvectors
        # e = e.real.double() # only take the real part, ignore imaginary part
        # v = v.real.double() # only take the real part, ignore imaginary part
        e = e.double() # only take the real part, ignore imaginary part
        v = v.double() # only take the real part, ignore imaginary part
        a = v.T @ torch.ones(n)
        b = v.T @ train_y
        mean = ((a/e) @ b) / ((a/e) @ a)
        d = v.T @ (train_y - mean)
        outputscale = (1./n * (d/e) @ d)
        noisescale = outputscale * torch.exp(log_lambda)
    # reset kernel lengthscale
    kernel._set_lengthscale(kernel.lengthscale()*(x_range[1] - x_range[0]))
    return (mean.item(), outputscale.item(), noisescale.item(), kernel)