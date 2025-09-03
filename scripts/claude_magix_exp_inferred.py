# import system package
import os
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
from optparse import OptionParser

# Add TensorFlow imports for npode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import types
tf.contrib = types.SimpleNamespace(
    distributions=types.SimpleNamespace(
        MultivariateNormalFullCovariance = tfp.distributions.MultivariateNormalFullCovariance,
        MultivariateNormalDiag           = tfp.distributions.MultivariateNormalDiag
    )
)

# Add torchdiffeq import for nrode
from torchdiffeq.adjoint import odeint_adjoint as odeint
from torchdiffeq import dynamic as nrode_dynamic

# import customize lib
import utils # experiment

# import magix
from magix.dynamic import nnMTModule
from magix.inference import FMAGI # inferred module

# import npode
from npode import npde_helper # inferred module

def main():
    # read in option
    usage = "usage:%prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-p", "--parameter", dest = "parameter_file_path",
                      type = "string", help = "path to the parameter file")
    parser.add_option("-r", "--result_dir", dest = "result_dir",
                      type = "string", help = "path to the result directory")
    parser.add_option("-s", dest = "random_seed", default = None,
                      type = "string", help = "random seed")
    (options, args) = parser.parse_args()

    # process parser information
    result_dir = options.result_dir
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)
    seed = options.random_seed
    if (seed is None):
        seed = (np.datetime64('now').astype('int')*104729) % 1e9
        seed = str(int(seed))

    # read in parameters
    parameters = utils.params()
    parameters.read(options.parameter_file_path)
    if (parameters.get('experiment','seed') is None):
        parameters.add('experiment','seed',seed)

    # read in data
    example = parameters.get('data','example')
    if (example is None):
        raise ValueError('parameter data: example must be provided!')
    data = np.loadtxt('data/321/%s.txt' %(example))
    tdata = data[:,0] # time
    xdata = data[:,1:] # component values
    no_comp = xdata.shape[1] # number of components
    # read in number of trainning points
    no_train = parameters.get('data','no_train')
    if (no_train is None):
        no_train = int((tdata.size-1)/2) + 1
        parameters.add('data','no_train',no_train)
    no_train = int(no_train)
    FIT_END = int((tdata.size-1)/2) + 1  # 161 when tdata.size == 321
    obs_idx = np.linspace(0,int((tdata.size-1)/2),no_train).astype(int)
    # obtain noise parameters
    noise = parameters.get('data','noise')
    noise = [float(x) for x in noise.split(',')]
    if (len(noise) != no_comp):
        if (len(noise) == 1):
            noise = [noise[0] for i in range(no_comp)]
        else:
            raise ValueError('noise parameters must have %d components!' %(no_comp))

    # read in experiment set up
    no_run = int(parameters.get('experiment','no_run'))
    # initialize/read in random seed for the data noise
    seed = int(parameters.get('experiment','seed'))
    exp_seed = utils.seed()
    exp_seed_file = parameters.get('experiment','seed_file')
    if (exp_seed_file is None):
        exp_seed.random(no_run, seed)
    else:
        exp_seed.load(exp_seed_file)
    exp_seed_file = os.path.join(result_dir, 'exp_seed.txt')
    exp_seed.save(exp_seed_file)
    parameters.add('experiment','seed_file',exp_seed_file)

    # read in model flag and model set-up
    # magix
    magix_run = parameters.get('magix','run')
    if (magix_run == 'yes'):
        magix_run = True
        # magix number of iterations
        magix_no_iter = int(parameters.get('magix','no_iteration'))
        # magix robust parameter
        magix_robust_eps = float(parameters.get('magix','robust_eps'))
        # magix hyperparameter update
        # magix_hyperparams_update = bool(int(parameters.get('magix','hyperparams_update')))
        # magix parameters
        magix_node = parameters.get('magix','hidden_node')
        magix_node = [no_comp] + [int(x) for x in magix_node.split(',')] + [no_comp]
        # set up heading for the output file
        magix_output_path = os.path.join(result_dir, 'magix.txt')
        magix_output = open(magix_output_path, 'w')
        magix_output.write('run,time')
        # heading for the output files
        for i in range(no_comp):
            for ptype in ['imputation','forecast','overall']:
                magix_output.write(',rmse_c%d_%s' %(i,ptype))
        magix_output.write('\n')
        magix_output.close()
    else:
        magix_run = False

    # npode
    npode_run = parameters.get('npode','run')
    if (npode_run == 'yes'):
        npode_run = True
        # npode number of iterations
        npode_no_iter = int(parameters.get('npode','no_iteration'))
        # set up heading for the output file
        npode_output_path = os.path.join(result_dir, 'npode.txt')
        npode_output = open(npode_output_path, 'w')
        npode_output.write('run,time')
        # heading for the output files
        for i in range(no_comp):
            for ptype in ['imputation','forecast','overall']:
                npode_output.write(',rmse_c%d_%s' %(i,ptype))
        npode_output.write('\n')
        npode_output.close()
    else:
        npode_run = False

    # neural ode
    nrode_run = parameters.get('nrode','run')
    if (nrode_run == 'yes'):
        nrode_run = True
        # neural ode number of iterations
        nrode_no_iter = int(parameters.get('nrode','no_iteration'))
        # neural ode parameters
        nrode_node = parameters.get('nrode','hidden_node')
        nrode_node = [int(x) for x in nrode_node.split(',')]
        nrode_node = [no_comp] + nrode_node + [no_comp]
        # set up heading for the output file
        nrode_output_path = os.path.join(result_dir, 'nrode.txt')
        nrode_output = open(nrode_output_path, 'w')
        nrode_output.write('run,time')
        # heading for the mse
        for i in range(no_comp):
            for ptype in ['imputation','forecast','overall']:
                nrode_output.write(',rmse_c%d_%s' %(i,ptype)) # reconstructed only
        nrode_output.write('\n')
        nrode_output.close()
    else:
        nrode_run = False

    # save parameters file
    parameters.save(result_dir)

    # run experiment
    for k in range(no_run):
        # data preprocessing
        obs = []
        np.random.seed(exp_seed.get(k)) # set random seed for noise
        for i in range(no_comp):
            tobs = tdata[obs_idx].copy()
            yobs = xdata[obs_idx,i].copy() + np.random.normal(0,noise[i],no_train)
            obs.append(np.hstack((tobs.reshape(-1,1),yobs.reshape(-1,1))))

        # Prepare data for npode/nrode (they need full observation vectors)
        tobs_full = tdata[obs_idx].copy()
        yobs_full = xdata[obs_idx,:].copy()
        for i in range(no_comp):
            yobs_full[:,i] = yobs_full[:,i] + np.random.normal(0,noise[i],no_train)

        # run models
        # magix
        if (magix_run):
            print('running magix...')
            # set random seed
            torch.manual_seed(exp_seed.get(k))
            # inference/learning
            start_time = time.time()
            # fOde = nnModule([no_comp, 512, no_comp]) # define nn dynamic
            fOde = nnMTModule(no_comp, 512) # define nn dynamic
            magix_model = FMAGI(obs,fOde,grid_size=161,interpolation_orders=3)
            # tinfer, xinfer = magix_model.map(
            #         max_epoch=magix_no_iter,
            #         learning_rate=1e-3, decay_learning_rate=True,
            #         robust=(magix_robust_eps>0), robust_eps=magix_robust_eps,
            #         hyperparams_update=0, dynamic_standardization=True,
            #         verbose=False, returnX=True)
            tinfer, xinfer = magix_model.map(max_epoch=magix_no_iter,
                    learning_rate=1e-3, decay_learning_rate=True,
                    hyperparams_update=False, dynamic_standardization=True,
                    verbose=True, returnX=True)
            end_time = time.time()
            
            # --- Compute inferred + forecast errors (no full reconstruction) ---
            # In-sample (imputation) error: compare inferred xinfer (161 x D) to ground truth first 161 points
            # Assumes tinfer aligns with tdata[:FIT_END] (default in FMAGI with grid_size=161)
            # Forecast: integrate forward from the last inferred state
            # Use t0 as last fit time, x0 as last inferred state
            t0_fore = tdata[FIT_END-1:FIT_END]       # shape (1,)
            x0_fore = xinfer[FIT_END-1:FIT_END, :]    # shape (1, D)
            tp_fore = tdata[FIT_END:]                 # shape (160,)

            # Predict only the forecasting segment
            t_pred, x_fore = magix_model.predict(
                tp=tp_fore,  # forecast times
                t0=t0_fore,  # initial time (last fit time)
                x0=x0_fore,  # initial state (last inferred state)
                random=False
            )

            run_time = end_time - start_time

            # Write RMSEs: imputation = inferred-fit; forecast = forecast segment; overall = concat
            magix_output = open(magix_output_path, 'a')
            magix_output.write('%d,%s' %(k,run_time))
            for i in range(no_comp):
                # Inferred (fit) error over first 161 points
                xi_err_fit = xinfer[:,i] - xdata[:FIT_END,i]
                rmse_fit = np.sqrt(np.mean(np.square(xi_err_fit)))
                magix_output.write(',%s' %(rmse_fit))

                # Forecast error over remaining 160 points
                xi_err_fore = x_fore[:,i] - xdata[FIT_END:,i]
                rmse_fore = np.sqrt(np.mean(np.square(xi_err_fore)))
                magix_output.write(',%s' %(rmse_fore))

                # Overall error across fit+forecast
                xi_err_all = np.concatenate([xi_err_fit, xi_err_fore], axis=0)
                rmse_all = np.sqrt(np.mean(np.square(xi_err_all)))
                magix_output.write(',%s' %(rmse_all))
            magix_output.write('\n')
            magix_output.close()

            # release memory
            del fOde
            del magix_model



        # npode
        if (npode_run):
            print('running npode...')
            # npode cannot handle (partial) missing data
            # remove missing data by taking the observation of first index
            npode_tobs = [tobs_full]
            npode_yobs = [yobs_full]
            tf.reset_default_graph()
            sess = tf.InteractiveSession()
            tf.set_random_seed(exp_seed.get(k))
            # inference/learning
            start_time = time.time()
            npode = npde_helper.build_model(sess, npode_tobs, npode_yobs, model='ode', sf0=1.0, ell0=np.ones(no_comp), W=6, ktype="id")
            x0, npode = npde_helper.fit_model(sess, npode, npode_tobs, npode_yobs, num_iter=npode_no_iter, print_every=100,eta=0.02, plot_=False)
            end_time = time.time()
            xrecon = npode.predict(x0,tdata).eval() # reconstruction
            sess.close()
            # release memory
            tf.get_default_graph().finalize()
            tf.reset_default_graph()
            # performance evaluation
            run_time = end_time - start_time
            npode_output = open(npode_output_path, 'a')
            npode_output.write('%d,%s' %(k,run_time))
            for i in range(no_comp):
                xi_error = xrecon[:,i] - xdata[:,i]
                xi_rmse_imputation = np.sqrt(np.mean(np.square(xi_error[:161])))
                npode_output.write(',%s' %(xi_rmse_imputation))
                xi_rmse_forecast = np.sqrt(np.mean(np.square(xi_error[161:])))
                npode_output.write(',%s' %(xi_rmse_forecast))
                xi_rmse_overall = np.sqrt(np.mean(np.square(xi_error)))
                npode_output.write(',%s' %(xi_rmse_overall))
            npode_output.write('\n')
            npode_output.close()
            del npode
            del sess

        # neural ode
        if (nrode_run):
            print('running nrode...')
            # neural ode cannot handle (partial) missing data
            # remove missing data by taking the observation of first index
            nrode_tobs = torch.tensor(tobs_full)
            nrode_yobs = torch.tensor(yobs_full).unsqueeze(1)
            # set random seed
            torch.manual_seed(exp_seed.get(k))
            # inference/learning
            start_time = time.time()
            func = nrode_dynamic.ODEFunc(nrode_node) # define neural network dynamic
            x0 = nrode_yobs[0]
            x0.requires_grad_(True)
            optimizer = torch.optim.RMSprop(list(func.parameters())+[x0], lr=1e-3)
            for itr in range(nrode_no_iter):
                optimizer.zero_grad()
                ypred = odeint(func, x0, nrode_tobs)
                loss = torch.mean(torch.abs(ypred - nrode_yobs))
                loss.backward()
                optimizer.step()
            end_time = time.time()
            with torch.no_grad():
                xrecon = odeint(func, x0, torch.tensor(tdata))
            xrecon = xrecon.detach().squeeze().numpy()
            run_time = end_time - start_time
            nrode_output = open(nrode_output_path, 'a')
            nrode_output.write('%d,%s' %(k,run_time))
            for i in range(no_comp):
                xi_error = xrecon[:,i] - xdata[:,i]
                xi_rmse_imputation = np.sqrt(np.mean(np.square(xi_error[:161])))
                nrode_output.write(',%s' %(xi_rmse_imputation))
                xi_rmse_forecast = np.sqrt(np.mean(np.square(xi_error[161:])))
                nrode_output.write(',%s' %(xi_rmse_forecast))
                xi_rmse_overall = np.sqrt(np.mean(np.square(xi_error)))
                nrode_output.write(',%s' %(xi_rmse_overall))
            nrode_output.write('\n')
            nrode_output.close()
            # release memory
            del func
            del x0


if __name__ == "__main__":
    main()