import torch
import gpytorch
import numpy as np
from .kernels.matern import MaternKernel
from .kernels.grid_interpolation import GridInterpolationKernel
from .integrator import RungeKutta

torch.set_default_dtype(torch.double)

class KISSGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid, interpolation_orders):
        super(KISSGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            GridInterpolationKernel(
                MaternKernel(), grid=grid, interpolation_orders=interpolation_orders
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FMAGI(object):
    def __init__(self, ys, dynamic, grid_size=201, interpolation_orders=3):
        self.grid_size = grid_size
        self.comp_size = len(ys)
        for i in range(self.comp_size):
            if (not torch.is_tensor(ys[i])):
                ys[i] = torch.tensor(ys[i]).squeeze()
        self.ys = ys
        self.fOde = dynamic
        self._kiss_gp_initialization(interpolation_orders=interpolation_orders)

    def _kiss_gp_initialization(self, interpolation_orders=3, training_iterations=100):
        tmin = self.ys[0][:,0].min()
        tmax = self.ys[0][:,0].max()
        for i in range(1, self.comp_size):
            tmin = torch.min(tmin, self.ys[i][:,0].min())
            tmax = torch.max(tmax, self.ys[i][:,0].max())
        spacing = (tmax - tmin) / (self.grid_size - 1)
        padding = int((interpolation_orders + 1) / 2)
        grid_bounds = (tmin - padding * spacing, tmax + padding * spacing)
        self.grid = torch.linspace(grid_bounds[0], grid_bounds[1], self.grid_size+2*padding)
        self.gp_models = []
        for i in range(self.comp_size):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            ti = self.ys[i][:,0]
            yi = self.ys[i][:,1]
            ti = ti[~torch.isnan(yi)]
            yi = yi[~torch.isnan(yi)]
            model = KISSGPRegressionModel(ti, yi, likelihood, self.grid, interpolation_orders)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # loss function
            for j in range(training_iterations):
                optimizer.zero_grad()
                output = model(ti)
                loss = -mll(output, yi)
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.gp_models.append(model)
        self.grid = self.grid[padding:-padding] # remove extended grid points

    def map(self, max_epoch=1000, 
            learning_rate=1e-3, decay_learning_rate=True,
            hyperparams_update=True, dynamic_standardization=False,
            kappa=1e3, verbose=False, returnX=False):
        gpmat = []
        u = torch.empty(self.grid_size, self.comp_size)
        x = torch.empty(self.grid_size, self.comp_size)
        dxdtGP = torch.empty(self.grid_size, self.comp_size)
        with torch.no_grad():
            for i in range(self.comp_size):
                ti = self.ys[i][:,0]
                model = self.gp_models[i]
                mean = model.mean_module.constant.item()
                outputscale = model.covar_module.outputscale.item()
                noisescale = model.likelihood.noise.item()
                nugget = noisescale / outputscale
                grid_kernel = model.covar_module.base_kernel
                base_kernel = grid_kernel.base_kernel
                # compute mean for grid points
                xi = model(self.grid).mean
                LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                LCinv = LC.inverse()
                ui = LCinv.matmul(xi-mean) / np.sqrt(outputscale)
                # compute uq for the grid points
                q = grid_kernel(ti,ti).add_jitter(nugget)._cholesky().inverse().matmul(grid_kernel(ti,self.grid))
                LU = (base_kernel(self.grid,self.grid)-q.t().matmul(q)).add_jitter(1e-6)._cholesky().mul(np.sqrt(outputscale))
                # compute gradient for grid points
                m = LCinv.matmul(base_kernel.dCdx2(self.grid,self.grid)).t()
                dxi = m.matmul(ui) * np.sqrt(outputscale)
                LKinv = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky().inverse()
                m = m.matmul(LCinv)
                # compute covariance for x|grid
                s = LCinv.matmul(grid_kernel(self.grid,ti))
                LQinv = (grid_kernel(ti,ti).add_jitter(nugget) - s.t().matmul(s)).add_jitter(1e-6)._cholesky().inverse()
                s = s.t().matmul(LCinv)
                # store information
                u[:,i] = ui
                x[:,i] = xi
                dxdtGP[:,i] = dxi
                gpmat.append({'LC':LC,'LCinv':LCinv,'m':m,'LKinv':LKinv,'s':s,'LQinv':LQinv,'LU':LU})

        # output standardization for fOde
        if (dynamic_standardization):
            dxdtGP_means = torch.mean(dxdtGP, axis=0)
            dxdtGP_stds = torch.std(dxdtGP, axis=0)
            self.fOde.update_output_layer(dxdtGP_means, dxdtGP_stds)

        # optimizer for u and theta
        state_optimizer = torch.optim.Adam([u], lr=learning_rate)
        theta_optimizer = torch.optim.Adam(self.fOde.parameters(), lr=learning_rate)
        theta_lambda = lambda epoch: (epoch+1) ** (-0.5)
        theta_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(theta_optimizer, lr_lambda=theta_lambda)

        # optimize initial theta
        # attach gradient for theta
        self.fOde.train()
        for param in self.fOde.parameters():
            param.requires_grad_(True)
        for _ in range(200):
            xr = x.clone()
            dxrdtOde = self.fOde(xr)
            theta_optimizer.zero_grad()
            lkh = torch.zeros(self.comp_size)
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                dxrdtError = gpmat[i]['LKinv'].matmul(dxrdtOde[:,i]-gpmat[i]['m'].matmul(xr[:,i]-mean))
                lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
            theta_loss = -torch.sum(lkh)
            theta_pn1 = dxrdtOde.norm(dim=1).square().mean()
            theta_pn2 = (torch.randn(self.comp_size) @ torch.func.vmap(torch.func.jacrev(self.fOde))(xr)).norm(dim=1).square().mean()
            theta_loss = theta_loss + kappa * (theta_pn1 + theta_pn2)
            theta_loss.backward()
            theta_optimizer.step()
        # detach theta gradient
        self.fOde.eval()
        for param in self.fOde.parameters():
            param.requires_grad_(False)

        for epoch in range(max_epoch):
            # optimize u (x after Cholesky decomposition)
            u.requires_grad_(True)
            for _ in range(1):
                state_optimizer.zero_grad()
                # reconstruct x
                x = torch.empty_like(u)
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])
                dxdtOde = self.fOde(x)
                lkh = torch.zeros((self.comp_size, 3))
                for i in range(self.comp_size):
                    mean = self.gp_models[i].mean_module.constant.item()
                    outputscale = self.gp_models[i].covar_module.outputscale.item()
                    # p(X[I] = x[I]) = P(U[I] = u[I])
                    lkh[i,0] = -0.5 * u[:,i].square().sum()
                    # p(Y[I] = y[I] | X[I] = x[I])
                    yiError = gpmat[i]['LQinv'].matmul(torch.nan_to_num(self.ys[i][:,1]-(mean+gpmat[i]['s'].matmul(x[:,i]-mean)),nan=0.0))
                    lkh[i,1] = -0.5/outputscale * yiError.square().sum()
                    # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                    dxidtError = gpmat[i]['LKinv'].matmul(dxdtOde[:,i]-gpmat[i]['m'].matmul(x[:,i]-mean))
                    lkh[i,2] = -0.5/outputscale * dxidtError.square().sum() / self.grid_size * yiError.size(0)
                state_loss = -torch.sum(lkh)  / self.grid_size
                state_loss.backward()
                state_optimizer.step()
            # detach gradient information
            u.requires_grad_(False)

            if (verbose and (epoch==0 or (epoch+1) % int(max_epoch/5) == 0)):
                print('%d/%d iteration: %.6f' %(epoch+1,max_epoch,state_loss.item()))

            # reconstruct x
            x = torch.empty_like(u)
            for i in range(self.comp_size):
                mean = self.gp_models[i].mean_module.constant.item()
                outputscale = self.gp_models[i].covar_module.outputscale.item()
                x[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(u[:,i])

            if ((epoch+1) < max_epoch):
                # update hyperparameter
                if (((epoch+1) % int(max_epoch/5) == 0) and hyperparams_update):
                    dxdtOde = self.fOde(x)
                    for i in range(self.comp_size):
                        ti = self.ys[i][:,0]
                        yi = self.ys[i][:,1]
                        xi = x[:,i]
                        model = self.gp_models[i]
                        model.train() 
                        model.likelihood.train()
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                        for _ in range(5):
                            optimizer.zero_grad()
                            # p(X[I] = x[I]) 
                            LC = model.covar_module.base_kernel.base_kernel(self.grid,self.grid)._cholesky()
                            LCinv = LC.inverse()
                            xiError = LCinv.matmul(xi-model.mean_module.constant)
                            lkh1 = -0.5 / model.covar_module.outputscale * xiError.square().sum()
                            lkh1 = lkh1 - 0.5 * self.grid_size * model.covar_module.outputscale.log() - LC.logdet()
                            # p(Y[I] = y[I] | X[I] = x[I])
                            nugget = model.covar_module.outputscale / model.likelihood.noise
                            s = LCinv.matmul(model.covar_module.base_kernel(self.grid,ti))
                            LQ = (model.covar_module.base_kernel(ti,ti).add_diagonal(nugget) - s.t().matmul(s))._cholesky()
                            s = s.t().matmul(LCinv)
                            yiError = LQ.inverse().matmul(torch.nan_to_num(yi-(model.mean_module.constant+s.matmul(xi-model.mean_module.constant)),nan=0.0))
                            lkh2 = -0.5 / model.covar_module.outputscale * yiError.square().sum()
                            lkh2 = lkh2 - 0.5 * self.grid_size * model.covar_module.outputscale.log() - LQ.logdet()
                            # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                            m = LCinv.matmul(model.covar_module.base_kernel.base_kernel.dCdx2(self.grid,self.grid)).t()
                            LK = (model.covar_module.base_kernel.base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky()
                            m = m.matmul(LCinv)
                            dxidtError = LK.inverse().matmul(dxdtOde[:,i]-m.matmul(x[:,i]-model.mean_module.constant))
                            lkh3 = - 0.5 / model.covar_module.outputscale * dxidtError.square().sum()
                            lkh3 = lkh3 - 0.5 * self.grid_size * model.covar_module.outputscale.log() - LK.logdet()
                            # loss = -(lkh1/self.grid_size + lkh2/ti.size(0) + lkh3/self.grid_size)
                            loss = -(lkh1 + lkh2 + lkh3/self.grid_size*ti.size(0)) / self.grid_size
                            loss.backward()
                            optimizer.step()
                        model.eval()
                        model.likelihood.eval()
                        self.gp_models[i] = model
                        # update gpmat
                        with torch.no_grad():
                            mean = model.mean_module.constant.item()
                            outputscale = model.covar_module.outputscale.item()
                            noisescale = model.likelihood.noise.item()
                            nugget = noisescale / outputscale
                            grid_kernel = model.covar_module.base_kernel
                            base_kernel = grid_kernel.base_kernel
                            # compute mean for the grid points
                            LC = base_kernel(self.grid,self.grid).add_jitter(1e-6)._cholesky()
                            LCinv = LC.inverse()
                            ui = LCinv.matmul(xi-mean) / np.sqrt(outputscale)
                            # compute uq for the grid points
                            q = grid_kernel(ti,ti).add_jitter(nugget)._cholesky().inverse().matmul(grid_kernel(ti,self.grid))
                            LU = (base_kernel(self.grid,self.grid)-q.t().matmul(q)).add_jitter(1e-6)._cholesky().mul(np.sqrt(outputscale))
                            # compute gradient for grid points
                            m = LCinv.matmul(base_kernel.dCdx2(self.grid,self.grid)).t()
                            LKinv = (base_kernel.d2Cdx1dx2(self.grid,self.grid)-m.matmul(m.t())).add_jitter(1e-6)._cholesky().inverse()
                            m = m.matmul(LCinv)
                            # assuming fixed noise, compute covariance for x|grid
                            s = LCinv.matmul(grid_kernel(self.grid,ti))
                            LQinv = (grid_kernel(ti,ti).add_jitter(nugget) - s.t().matmul(s)).add_jitter(1e-6)._cholesky().inverse()
                            s = s.t().matmul(LCinv)
                            # store information
                            u[:,i] = ui
                            gpmat[i] = {'LC':LC,'LCinv':LCinv,'m':m,'LKinv':LKinv,'s':s,'LQinv':LQinv,'LU':LU}

                self.fOde.train()
                for param in self.fOde.parameters():
                    param.requires_grad_(True)
                for _ in range(1):
                    xr = x.clone()
                    dxrdtOde = self.fOde(xr)
                    theta_optimizer.zero_grad()
                    lkh = torch.zeros(self.comp_size)
                    for i in range(self.comp_size):
                        mean = self.gp_models[i].mean_module.constant.item()
                        outputscale = self.gp_models[i].covar_module.outputscale.item()
                        dxrdtError = gpmat[i]['LKinv'].matmul(dxrdtOde[:,i]-gpmat[i]['m'].matmul(xr[:,i]-mean))
                        lkh[i] = -0.5/outputscale * dxrdtError.square().mean()
                    theta_loss = -torch.sum(lkh)
                    theta_pn1 = dxrdtOde.norm(dim=1).square().mean()
                    theta_pn2 = (torch.randn(self.comp_size) @ torch.func.vmap(torch.func.jacrev(self.fOde))(xr)).norm(dim=1).square().mean()
                    theta_loss = theta_loss + kappa * (theta_pn1 + theta_pn2)
                    theta_loss.backward()
                    theta_optimizer.step()
                if (decay_learning_rate):
                    theta_lr_scheduler.step()
                # detach theta gradient
                self.fOde.eval()
                for param in self.fOde.parameters():
                    param.requires_grad_(False)

        if (returnX):
            return (self.grid.numpy(), x.numpy())

    def predict(self, tp, t0, x0, random=False, method="ode", **params):
        # obtain prediction by ode numerical integration / gaussian process map
        if (method=="ode" and random):
            self.fOde.train() # training mode with dropout
        else:
            self.fOde.eval() # evaluation mode without dropout
            for param in self.fOde.parameters():
                param.requires_grad_(False)
        # call integrater
        itg = RungeKutta(self.fOde)
        # preprocess input
        if (not torch.is_tensor(t0)):
            t0 = torch.tensor(t0).reshape(-1,)
        if (not torch.is_tensor(tp)):
            tp = torch.tensor(tp).reshape(-1,)
        if (not torch.is_tensor(x0)):
            x0 = torch.tensor(x0).squeeze()
            if (x0.ndimension()==1):
                x0 = x0.unsqueeze(0)
        if (method=="ode"):
            xp = itg.forward(x0[-1,:], torch.cat((t0[-1].unsqueeze(0),tp)), **params)[1:,:]
        else:
            # gaussian process map method
            # optimize for computing xp
            xp = torch.empty(tp.size(0), self.comp_size)
            for j in range(tp.size(0)):
                ts = torch.cat((t0,tp[:(j+1)]))
                xs = torch.cat((x0,xp[:(j+1),:]))
                # obtain intial estimation by ode integration
                xs[-1,:] = itg.forward(xs[-2,:], ts[-2:], **params)[-1,:].detach()
                # get hyperparameters
                with torch.no_grad():
                    gpmat = []
                    us = torch.empty_like(xs)
                    for i in range(self.comp_size):
                        model = self.gp_models[i]
                        mean = model.mean_module.constant.item()
                        outputscale = model.covar_module.outputscale.item()
                        grid_kernel = model.covar_module.base_kernel
                        base_kernel = grid_kernel.base_kernel
                        # compute satte information
                        LC = base_kernel(ts,ts).add_jitter(1e-6)._cholesky()
                        LCinv = LC.inverse()
                        us[:,i] = LCinv.matmul(xs[:,i]-mean) / np.sqrt(outputscale)
                        # compute gaussian process gradient
                        m = LCinv.matmul(base_kernel.dCdx2(ts,ts)).t()
                        LKinv = (base_kernel.d2Cdx1dx2(ts,ts)-m.matmul(m.t())).add_jitter(1e-6)._cholesky().inverse()
                        m = m.matmul(LCinv)
                        gpmat.append({'LC':LC,'m':m,'LKinv':LKinv})
                # perform optimization
                uo = us[:-1,:].detach().clone() # fixed all prior points
                un = us[-1:,:].detach().clone() # only optimize for the next
                un.requires_grad_(True)
                optimizer = torch.optim.Adam([un], lr=1e-2)
                for _ in range(10):
                    optimizer.zero_grad()
                    # reconstruct x
                    us = torch.cat((uo,un))
                    xs = torch.empty_like(us)
                    for i in range(self.comp_size):
                        model = self.gp_models[i]
                        mean = model.mean_module.constant.item()
                        outputscale = model.covar_module.outputscale.item()
                        xs[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(us[:,i])
                    dxdtOde = self.fOde(xs)
                    lkh = torch.zeros((self.comp_size, 2))
                    for i in range(self.comp_size):
                        model = self.gp_models[i]
                        mean = model.mean_module.constant.item()
                        outputscale = model.covar_module.outputscale.item()
                        # p(X[I] = x[I]) = P(U[I] = u[I])
                        lkh[i,0] = -0.5 * us[:,i].square().sum()
                        # p(X'[I]=f(x[I],theta)|X(I)=x(I))
                        dxidtError = gpmat[i]['LKinv'].matmul(dxdtOde[:,i]-gpmat[i]['m'].matmul(xs[:,i]-mean))
                        lkh[i,1] = -0.5/outputscale * dxidtError.square().sum()
                    loss = -torch.sum(lkh)
                    loss.backward()
                    optimizer.step()
                # reconstruct x
                un.requires_grad_(False)
                us = torch.cat((uo,un))
                xs = torch.empty_like(us)
                for i in range(self.comp_size):
                    model = self.gp_models[i]
                    mean = model.mean_module.constant.item()
                    outputscale = model.covar_module.outputscale.item()
                    xs[:,i] = mean + np.sqrt(outputscale) * gpmat[i]['LC'].matmul(us[:,i])
                xp[j,:] = xs[-1,:]                    
        t = torch.cat((t0,tp))
        x = torch.cat((x0,xp))
        return (t.numpy(), x.numpy())