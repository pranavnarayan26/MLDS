import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax


#this thing is a constructor, apparently; and constructors in python apparently always start with a reference to the current instance
class Func(eqx.Module):
    mlp: eqx.nn.MLP # MLP is apparently just the standard feedforward neural network; takes in an input layer, goes through some hidden layers, and has an output layer

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth, #depth of neural ode / # of layers
            activation=jnn.tanh, #activation function applied to each layer; 
            #softplus is ln(1+e^x), which approaches 0 for large negatives, and approaches linearity for large positives
            key=key, #used to randomize initial parameter
        )

    def __call__(self, t, y, args): #allows object name to be used as a function ex. self(1, 2, [idk]) would call this (and probably result in an infinite loop, but still)
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(), #the method by which the ODE is being solved
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts), max_steps = 1000000, 
        )
        return solution.ys

def _get_data(ts, *, key):
    #data_key, model_key, loader_key = jr.split(key, 3)
    def f(t, state, args):
        G = 1
        M = 1
        mu = G * M

        r = jnp.linalg.norm(jnp.asarray(state[:3]))

        a = -mu * jnp.array(state[:3], float) / r**3

        return [state[3], state[4], state[5], a[0], a[1], a[2]]

    term = diffrax.ODETerm(f)
    solver = diffrax.Tsit5()
    
    inclination = 0
    G = 1
    M = 1
    r = 1
    a = r
    mu = G * M
    T = 2 * 2 * jnp.pi * jnp.sqrt(a**3/mu) #formula for period of an orbit = 2 * jnp.pi * jnp.sqrt(a**3/mu)
    v = jnp.sqrt(2 * mu/r - mu/a) #formula for velocity of an orbit = jnp.sqrt(2 * mu/r - mu/a)


    y0 = [0, r * jnp.cos(inclination), -r * jnp.sin(inclination), v, 0, 0] 
    ts = jnp.linspace(0, T, 100) 
    saveat = diffrax.SaveAt(ts=ts)

    sol = diffrax.diffeqsolve(term, solver, ts[0], ts[-1], dt0 = 0.1, y0=y0, saveat=saveat, max_steps=1000000)
    ys = sol.ys #i have no idea why this is necessary, namely the bit with the key
    #print(jnp.shape(ys))
    ys = jnp.swapaxes(jnp.array(ys), 0, 1)
    return ts, ys


def get_data(dataset_size, *, key):
    G = 1
    M = 1
    mu = G * M
    T = 2 * 2 * jnp.pi * jnp.sqrt(1**3/mu)
    ts = jnp.linspace(0, T, 100) 
    key = jr.split(key, dataset_size)
    ts, ys1 = _get_data(ts, key=key)
    #print(jnp.shape(ys1))
    ys2 = jax.vmap(lambda key: ys1)(key)
    ys = jnp.array(ys2)
    #print("still working after get_data")
    return ts, ys2

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size) #arange returns evenly spaced values within a given interval; in this case, 0 to dataset_size exclusive, integer values only
    #print("still working before infinite loop in dataloader")
    while True: #apparently this infinite loop is to continuously supply data?
        perm = jr.permutation(key, indices) #returns a shuffled version of indices
        (key,) = jr.split(key, 1) #splits key into two, stores the first half?
        start = 0
        end = batch_size #32
        #print("still working before internal loop in dataloader")
        while end < dataset_size: #end < 1000
            batch_perm = perm[start:end] #takes end (32) elements from the indices
            #for array in arrays:
             #   print(array[batch_perm])
            yield tuple(array[batch_perm] for array in arrays) #returns tuples of 
            #print(array[batch_perm])
            start = end
            end = start + batch_size

def main(
    dataset_size=256,
    batch_size=32,
    lr_strategy=(9e-3, 9e-3),
    steps_strategy=(1000, 390),
    length_strategy=(0.2, 1),
    width_size=16,
    depth=2,
    seed=5678,
    plot=True,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    #print(key)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = jnp.shape(ys)
    #print(ys.shape)
    #print(length_size)
    #print(data_size)

    model = NeuralODE(data_size, width_size, depth, key=model_key)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        #print(yi[:, 0])
        y_pred = jax.vmap(model, in_axes= (None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        #print("working before grad_loss")
        loss, grads = grad_loss(model, ti, yi)
        #print("working after grad_loss")
        updates, opt_state = optim.update(grads, opt_state)
        #print("working after optim.update")
        model = eqx.apply_updates(model, updates)
        #print(model)
        return loss, model, opt_state

    #this whole loop runs twice - the three strategy values have 2 elements each
    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy): #combines elements of each list into tuples
        losses = []
        stepper = []
        optim = optax.adabelief(lr) #initializes optimizer called adabelief, with lr being "learning rate"?
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        #print(model)
        #based on context, and considering is_inexact_array is a boolean function that returns true if an element is an inexact array(whatever that means)
        # it's possible filter just identifies those elements for which is_inexact_array returns true
        _ts = jnp.array(ts[: int(length_size * length)]) #this seems to define how far into the dataset training occurs, because length is first 0.1 then 1
        #also this is splitting ts into its first (length_size * length) elements, for clarity
        _ys = ys[:, : int(length_size * length)] #same as above except with ys
        #this loop should run steps (500) times
        #print(_ys)
    
        #hm = dataloader((_ys,), batch_size, key=loader_key)
        #print(hm)
        
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ): 
            #print("main's internal for loop (for training) works")
            start = time.time()
            #print(jnp.array(yi))
            loss, model, opt_state = make_step(_ts, jnp.array(yi), model, opt_state)
            #print(f"Loss: {loss}")
            end = time.time()
            #if (step % print_every) == 0 or step == steps:
            print(f"Step: {step + 1}, Loss: {loss}, Computation time: {end - start}")
            losses.append(loss)
            stepper.append(step)
        
    


    if plot:
        
        plt.figure(1)
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real x")
        plt.plot(ts, ys[0, :, 1], c="crimson", label = "Real y")
        plt.plot(ts, ys[0, :, 2], c="yellow", label = "Real z")
        model_y = model(ts, ys[0, 0])
        plt.plot(ts, model_y[:, 0], c="lime", label="Model x")
        plt.plot(ts, model_y[:, 1], c="magenta", label = "Model y")
        plt.plot(ts, model_y[:, 2], c="black", label = "Model z")
        """model_y = model(ts, ys[0, :, 0])
        print(jnp.shape(model_y))
        plt.plot(ts, model_y[0, :, 0], c="magenta", label="Model x")
        plt.plot(ts, model_y[0, :, 1], c="black", label = "Model y")
        plt.plot(ts, model_y[0, :, 2], c="green", label = "Model z")"""

        plt.legend()
        plt.tight_layout()
        plt.savefig("dimensionalized_orbit2.png")

        plt.figure(2)
        plt.plot(stepper[100:], losses[100:], label = "Loss")
        plt.tight_layout()
        plt.savefig("loss-step.png")
        #plt.show()

    return ts, ys, model

ts, ys, model = main()