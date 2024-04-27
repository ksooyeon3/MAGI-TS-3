import torch
import numpy as np
from . import gpm
from . import integrator

torch.set_default_dtype(torch.double)

class magix(object):
    def __init__(self, y, t, dynamic):
        '''
        Constructor of the inference module
        Args:
            y: 2d observations, missing data is nan
            t: 1d observation time
        '''
        # preprocess y
        if (not torch.is_tensor(y)):
            y = torch.tensor(y)
        y = y.double().squeeze()
        if (len(y.size()) < 2):
            y = y.reshape(-1,1)
        self.y = y
        self.n, self.p = y.size()
        # preprocess t
        if (not torch.is_tensor(t)):
            t = torch.tensor(t)
        t = t.double().squeeze()
        self.t = t
        # dynamic initialization
        self.fOde = dynamic
        # gaussian process fitting
        self.fitGP()

    def fitGP(self):
        '''
        GP preprocessing of the Input Data
        '''
        self.gp_models = []
        for i in range(self.p):
            aidx = ~torch.isnan(self.y[:,i]) # non-missing data index
            ti = self.t[aidx]
            yi = self.y[aidx,i]
            mean, outputscale, noisescale, kernel = gpm.GPTrain(ti, yi)
            self.gp_models.append({
                'aidx':aidx, 
                'mean':mean,
                'outputscale':outputscale,
                'noisescale':noisescale,
                'kernel':kernel
            })

    def robustMAP(self, max_epoch=1000, eps=0.05, verbose=False, returnX=False):
        '''
        Robust MAP inference of ODE parameters
        '''

        # obtain features from GP Models
        gpcov = []
        logvar = torch.empty(self.p).double()
        u = torch.empty_like(self.y).double()
        x = torch.empty_like(self.y).double()
        dxdtGP = torch.empty_like(self.y).double()
        for i in range(self.p):
            # get observation data
            aidx = self.gp_models[i]['aidx']
            yi = self.y[aidx,i]
            ti = self.t[aidx]
            # get hyperparameters
            kernel = self.gp_models[i]['kernel']
            mean = self.gp_models[i]['mean']
            outputscale = self.gp_models[i]['outputscale']
            noisescale = self.gp_models[i]['noisescale']
            # obtain state information
            Ctz = kernel.C(self.t, ti)
            Czz = kernel.C(ti) + (noisescale/outputscale) * torch.eye(ti.size(0))
            xmean = mean + Ctz @ torch.linalg.inv(Czz) @ (yi - mean)
            Ctt = kernel.C(self.t) + 1e-4 * torch.eye(self.n)
            LC = torch.linalg.cholesky(Ctt)
            LCinv = torch.linalg.inv(LC)
            Cinv = LCinv.T @ LCinv
            xcov = Ctt - Ctz @ torch.linalg.inv(Czz) @ Ctz.T
            LU = torch.linalg.cholesky(xcov)
            # obtain derivative information
            dCdt1 = kernel.dC_dx1(self.t)
            dCdt2 = kernel.dC_dx2(self.t)
            d2Cdt1dt2 = kernel.d2C_dx1dx2(self.t)
            m = dCdt1 @ Cinv
            K = d2Cdt1dt2 - m @ dCdt2 + 1e-4 * torch.eye(self.n)
            Kinv = torch.linalg.inv(K)
            # store information
            logvar[i] = torch.log(torch.tensor(noisescale))
            x[:,i] = xmean
            u[:,i] = LCinv @ (xmean - mean) / np.sqrt(outputscale)
            dxdtGP[:,i] = m @ (xmean - mean)
            gpcov.append({'LC':LC,'LU':LU,'m':m,'Kinv':Kinv})

        # compute standardization parameter for dxdtGP
        dxdtGP_means = torch.mean(dxdtGP, axis=0).detach()
        dxdtGP_stds = torch.std(dxdtGP, axis=0).detach()
        self.fOde.update_output_layer(dxdtGP_means, dxdtGP_stds)

        # optimizer for u and theta
        state_optimizer = torch.optim.Adam([u,logvar], lr=1e-3)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr=1e-3)
        theta_lambda = lambda epoch: (epoch+1) ** (-0.5)
        theta_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(theta_optimizer, lr_lambda=theta_lambda)

        # optimize initial theta
        # attach gradient for theta
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        for tt in range(200):
            # identify adversarial path
            delta = torch.zeros_like(x, requires_grad = True)
            xadv = torch.empty_like(x)
            for i in range(self.p):
                xadv[:,i] = x[:,i] + gpcov[i]['LU'] @ delta[:,i]
            dxadvdtOde = self.fOde(xadv)
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.p)
            for i in range(self.p):
                mean = self.gp_models[i]['mean']
                outputscale = self.gp_models[i]['outputscale']
                dxadvdtError = dxadvdtOde[:,i] - (gpcov[i]['m'] @ (xadv[:,i] - mean))
                lkh[i] = -0.5/outputscale * dxadvdtError @ gpcov[i]['Kinv'] @ dxadvdtError 
            theta_loss = -torch.sum(lkh)
            theta_loss.backward()
            xadv = torch.empty_like(x)
            for i in range(self.p):
                # gradient ascent to find adversarial path
                xadv[:,i] = x[:,i] + gpcov[i]['LU'] @ (eps * delta.grad.data.sign()[:,i])
            # optimize theta over the adversarial path
            dxadvdtOde = self.fOde(xadv)
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.p)
            for i in range(self.p):
                mean = self.gp_models[i]['mean']
                outputscale = self.gp_models[i]['outputscale']
                dxadvdtError = dxadvdtOde[:,i] - (gpcov[i]['m'] @ (xadv[:,i] - mean))
                lkh[i] = -0.5/outputscale * dxadvdtError @ gpcov[i]['Kinv'] @ dxadvdtError 
            theta_loss = -torch.sum(lkh)
            theta_loss.backward()
            theta_optimizer.step()
        # detach theta gradient
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        for epoch in range(max_epoch):
            # optimize u (x after Cholesky decomposition)
            u.requires_grad_(True)
            logvar.requires_grad_(True)
            for st in range(1):
                state_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty_like(u).double()
                dxdtGP = torch.empty_like(u).double()
                for i in range(self.p):
                    mean = self.gp_models[i]['mean']
                    outputscale = self.gp_models[i]['outputscale']
                    x[:,i] = mean + np.sqrt(outputscale) * gpcov[i]['LC'] @ u[:,i]
                    dxdtGP[:,i] = gpcov[i]['m'] @ (x[:,i] - mean)
                dxdtOde = self.fOde(x)
                lkh = torch.zeros((self.p, 3))
                for i in range(self.p):
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * torch.mean(torch.square(u[:,i]))
                    # p(Y[I] = y[I] | X[I] = x[I])
                    aidx = self.gp_models[i]['aidx']
                    noisescale = torch.exp(logvar[i])
                    lkh[i,1] = -0.5/noisescale * torch.sum(torch.square(x[aidx,i]-self.y[aidx,i])) / self.n
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    outputscale = self.gp_models[i]['outputscale']
                    dxdtError = dxdtOde[:,i] - dxdtGP[:,i]
                    lkh[i,2] = -0.5/outputscale * dxdtError @ gpcov[i]['Kinv'] @ dxdtError / self.n
                state_loss = -torch.sum(lkh)
                state_loss.backward()
                state_optimizer.step()
            # detach gradient information
            u.requires_grad_(False) 
            logvar.requires_grad_(False)

            if (verbose and (epoch==0 or (epoch+1) % int(max_epoch/5) == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,max_epoch,state_loss.item()))

            # reconstruct x
            x = torch.empty_like(u).double()
            for i in range(self.p):
                mean = self.gp_models[i]['mean']
                outputscale = self.gp_models[i]['outputscale']
                x[:,i] = mean + np.sqrt(outputscale) * gpcov[i]['LC'] @ u[:,i]

            if (epoch+1 < max_epoch):
                # decay adversarial rate
                eps = eps * ((epoch+1) ** (-0.5))
                # optimize over theta 
                # attach gradient for theta
                for param in self.fOde.parameters():
                    param.requires_grad_(True)
                for tt in range(1):
                    # identify adversarial path
                    delta = torch.zeros_like(x, requires_grad = True)
                    xadv = torch.empty_like(x)
                    for i in range(self.p):
                        xadv[:,i] = x[:,i] + gpcov[i]['LU'] @ delta[:,i]
                    dxadvdtOde = self.fOde(xadv)
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.p)
                    for i in range(self.p):
                        mean = self.gp_models[i]['mean']
                        outputscale = self.gp_models[i]['outputscale']
                        dxadvdtError = dxadvdtOde[:,i] - (gpcov[i]['m'] @ (xadv[:,i] - mean))
                        lkh[i] = -0.5/outputscale * dxadvdtError @ gpcov[i]['Kinv'] @ dxadvdtError 
                    theta_loss = -torch.sum(lkh)
                    theta_loss.backward()
                    xadv = torch.empty_like(x)
                    for i in range(self.p):
                        # gradient ascent to find adversarial path
                        xadv[:,i] = x[:,i] + gpcov[i]['LU'] @ (eps * delta.grad.data.sign()[:,i])
                    # optimize theta over the adversarial path
                    dxadvdtOde = self.fOde(xadv)
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.p)
                    for i in range(self.p):
                        mean = self.gp_models[i]['mean']
                        outputscale = self.gp_models[i]['outputscale']
                        dxadvdtError = dxadvdtOde[:,i] - (gpcov[i]['m'] @ (xadv[:,i] - mean))
                        lkh[i] = -0.5/outputscale * dxadvdtError @ gpcov[i]['Kinv'] @ dxadvdtError 
                    theta_loss = -torch.sum(lkh)
                    theta_loss.backward()
                    theta_optimizer.step()
                theta_lr_scheduler.step()
                # detach theta gradient
                for param in self.fOde.parameters():
                    param.requires_grad_(False)

        for i in range(self.p):
            self.gp_models[i]['noisescale'] = torch.exp(logvar[i]).item()

        if (returnX):
            return (x.numpy())

    def predict(self, x0, ts, **params):
        # obtain initialization by numerical integration
        itg = integrator.RungeKutta(self.fOde)
        ts = torch.tensor(ts).double().squeeze()
        x = itg.forward(x0, ts, **params)
        return (x.numpy())
