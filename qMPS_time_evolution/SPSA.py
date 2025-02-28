import numpy as np
from scipy.optimize import OptimizeResult
from classical import expectation

### ADAPTED FROM NOISYOPT
def minimizeSPSA(func, x0, args=(), bounds=None, niter=100, paired=True,
                 a=1.0, alpha=0.602, c=1.0, gamma=0.101,
                 disp=False, callback=None, restart_point=0):
    """
    Minimization of an objective function by a simultaneous perturbation
    stochastic approximation algorithm.
    This algorithm approximates the gradient of the function by finite differences
    along stochastic directions Deltak. The elements of Deltak are drawn from
    +- 1 with probability one half. The gradient is approximated from the 
    symmetric difference f(xk + ck*Deltak) - f(xk - ck*Deltak), where the evaluation
    step size ck is scaled according ck =  c/(k+1)**gamma.
    The algorithm takes a step of size ak = a/(0.01*niter+k+1)**alpha along the
    negative gradient.
    
    See Spall, IEEE, 1998, 34, 817-823 for guidelines about how to choose the algorithm's
    parameters (a, alpha, c, gamma).
    Parameters
    ----------
    func: callable
        objective function to be minimized:
        called as `func(x, *args)`,
        if `paired=True`, then called with keyword argument `seed` additionally
    x0: array-like
        initial guess for parameters 
    args: tuple
        extra arguments to be supplied to func
    bounds: array-like
        bounds on the variables
    niter: int
        number of iterations after which to terminate the algorithm
    paired: boolean
        calculate gradient for same random seeds
    a: float
       scaling parameter for step size
    alpha: float
        scaling exponent for step size
    c: float
       scaling parameter for evaluation step size
    gamma: float
        scaling exponent for evaluation step size 
    disp: boolean
        If true, output 100 status updates during the optimization,
        or every step if niter<100.
    callback: callable
        called after each iteration, as callback(xk), where xk are the current parameters
    restart_point: int
        how many updates in the restart from, called if there has been a failure while optimising, optimiser restarts from parameters it had reached before failure point
    Returns
    -------
    `scipy.optimize.OptimizeResult` object
    """
    A = 0.01 * niter

    if bounds is not None:
        bounds = np.asarray(bounds)
        project = lambda x: np.clip(x, bounds[:, 0], bounds[:, 1])

    if args is not None:
        # freeze function arguments
        def funcf(x, **kwargs):
            return func(x, *args, **kwargs)


    N = len(x0)
    x = x0
    for k in range(restart_point,niter):
        ak = a/(k+1.0+A)**alpha
        ck = c/(k+1.0)**gamma
        Deltak = np.random.choice([-1, 1], size=N)
        fkwargs = dict()
        if paired:
            # upper bound needs to be set to signed 32-bit integer
            # see https://github.com/numpy/numpy/issues/4085#issuecomment-29570567
            fkwargs['seed'] = np.random.randint(0, np.iinfo(np.int32).max)
        if bounds is None:
            grad = (funcf(x + ck*Deltak, **fkwargs) - funcf(x - ck*Deltak, **fkwargs)) / (2*ck*Deltak)
            x -= ak*grad
        else:
            # ensure evaluation points are feasible
            xplus = project(x + ck*Deltak)
            xminus = project(x - ck*Deltak)
            grad = (funcf(xplus, **fkwargs) - funcf(xminus, **fkwargs)) / (xplus-xminus)
            x = project(x - ak*grad)
        # print status updates every 100th iteration if disp=True
        if disp and (k % max([1, niter//100])) == 0:
            print(x)
        if callback is not None:
            callback(x)
    message = 'terminated after reaching max number of iterations'
    return OptimizeResult(fun=expectation(args[0],x,0.2,0.2), x=x, nit=niter, nfev=2*niter, message=message, success=True)


