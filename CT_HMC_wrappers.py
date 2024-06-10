import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp

from functools import partial


def augmented_model(
    model,
    *model_args,
    temp_ctrl_name='temp_ctrl',
    inv_temp_name='inv_temp',
    **model_kwargs
):
    '''Add the temperature control variable and the inverse temperature to the model'''
    def temp_model():
        temp_ctrl = numpyro.sample(temp_ctrl_name, dist.Logistic(loc=0.0, scale=1.0))
        numpyro.deterministic(inv_temp_name, jax.nn.sigmoid(temp_ctrl))
        model(*model_args, **model_kwargs)
    return temp_model


def split_density(
    model,
    base_dist,
    guide_order,
    params,
    temp_ctrl_name='temp_ctrl',
    inv_temp_name='inv_temp'
):
    params_samples = jnp.hstack([
        params[key].ravel() for key in guide_order
    ])

    # guide works with unconstrained parameters the same as potential energy
    log_base = base_dist.log_prob(params_samples).sum()
    substituted_model = numpyro.handlers.substitute(
        model, substitute_fn=partial(numpyro.infer.util._unconstrain_reparam, params)
    )
    log_joint, model_trace = numpyro.infer.util.log_density(
        substituted_model, (), {}, {}
    )

    log_temp = model_trace[temp_ctrl_name]['fn'].log_prob(
        model_trace[temp_ctrl_name]['value']
    )

    log_target = log_joint - log_temp
    inv_temp = model_trace[inv_temp_name]['value']
    return log_target, log_base, inv_temp, log_temp


def potential_energy(
    model,
    base_dist,
    neg_elbo,
    guide_order,
    params,
    temp_ctrl_name='temp_ctrl',
    inv_temp_name='inv_temp'
):
    '''Calculate the augmented potential energy (equation 14)'''
    log_target, log_base, inv_temp, log_temp = split_density(
        model,
        base_dist,
        guide_order,
        params,
        temp_ctrl_name=temp_ctrl_name,
        inv_temp_name=inv_temp_name
    )
    # flip sign of everything to match numpyro convention
    return -(inv_temp * (log_target + neg_elbo) + (1 - inv_temp) * log_base + log_temp)
    # return (inv_temp * (neg_elbo - log_joint) + (inv_temp - 1) * (log_base + log_temp))


def delta(
    model,
    base_dist,
    neg_elbo,
    guide_order,
    params,
    temp_ctrl_name='temp_ctrl',
    inv_temp_name='inv_temp'
):
    '''Calculate the delta used in the weights (paragraph below equation 17)'''
    log_target, log_base, _, _ = split_density(
        model,
        base_dist,
        guide_order,
        params,
        temp_ctrl_name=temp_ctrl_name,
        inv_temp_name=inv_temp_name
    )
    return -neg_elbo - log_target + log_base


def postprocessing(
    model,
    base_dist,
    neg_elbo,
    guide_order,
    params,
    temp_ctrl_name='temp_ctrl',
    inv_temp_name='inv_temp',
    delta_name='delta',
    w0_name='w0',
    w1_name='w1'
):
    '''Add deterministic variables to the samples in the model'''
    processed_params = numpyro.infer.util.constrain_fn(
        model,
        (),
        {},
        params,
        return_deterministic=True
    )
    delta_value = delta(
        model,
        base_dist,
        neg_elbo,
        guide_order,
        params,
        temp_ctrl_name=temp_ctrl_name,
        inv_temp_name=inv_temp_name
    )
    # Equation 20
    w0 = jnp.where(delta_value == 0, 1.0, -delta_value / jax.lax.expm1(-delta_value))
    w1 = jnp.where(delta_value == 0, 1.0, delta_value / jax.lax.expm1(delta_value))
    processed_params[delta_name] = delta_value
    processed_params[w0_name] = w0
    processed_params[w1_name] = w1
    return processed_params


def sample_init_params(
    model,
    model_args,
    model_kwargs,
    guide,
    rng_key,
    guide_params,
    num_chains,
    temp_ctrl_name='temp_ctrl'
):
    init_params = guide.sample_posterior(rng_key, guide_params, sample_shape=(num_chains,))
    unconstrained_init_params = unconstrained_vmap(
        model,
        model_args,
        model_kwargs,
        init_params,
        jnp.arange(num_chains)
    )
    unconstrained_init_params[temp_ctrl_name] = jnp.zeros(num_chains) - 20.0
    return unconstrained_init_params


def wrap_model(
    model,
    model_args,
    model_kwargs,
    guide,
    svi_results=None,
    neg_elbo=None,
    guide_params=None,
    temp_ctrl_name='temp_ctrl',
    inv_temp_name='inv_temp',
    delta_name='delta',
    w0_name='w0',
    w1_name='w1'
):
    if svi_results is not None:
        svi_tail = svi_results.losses.shape[0] // 5
        # average SVI loss in the last 20%
        neg_elbo = svi_results.losses[-svi_tail:].mean()
        base_dist = guide.get_posterior(params=svi_results.params)
    else:
        neg_elbo = neg_elbo
        base_dist = guide.get_posterior(params=guide_params)
    guide_order = guide._init_locs.keys()

    temp_model = augmented_model(
        model,
        *model_args,
        temp_ctrl_name=temp_ctrl_name,
        inv_temp_name=inv_temp_name,
        **model_kwargs
    )
    potential_energy_fn = partial(
        potential_energy,
        temp_model,
        base_dist,
        neg_elbo,
        guide_order,
        temp_ctrl_name=temp_ctrl_name,
        inv_temp_name=inv_temp_name
    )
    postprocessing_fn = partial(
        postprocessing,
        temp_model,
        base_dist,
        neg_elbo,
        guide_order,
        temp_ctrl_name=temp_ctrl_name,
        inv_temp_name=inv_temp_name,
        delta_name=delta_name,
        w0_name=w0_name,
        w1_name=w1_name
    )
    sample_init_fn = partial(
        sample_init_params,
        model,
        model_args,
        model_kwargs,
        guide,
        temp_ctrl_name=temp_ctrl_name
    )
    return temp_model, potential_energy_fn, sample_init_fn, postprocessing_fn, neg_elbo


def nlogZ(neg_elbo, samples, w0_name='w0', w1_name='w1'):
    w0_sum = samples[w0_name].sum()
    w1_sum = samples[w1_name].sum()
    return neg_elbo - jnp.log(w1_sum) + jnp.log(w0_sum)


def re_center_guide(
    model,
    model_args,
    model_kwargs,
    mcmc,
    guide,
    neg_elbo,
    w0_name='w0',
    w1_name='w1'
):
    '''Recenter guide based on mcmc samples from previous run'''
    sample_sites = [
        site['name']
        for site in guide.prototype_trace.values()
        if ((site['type'] == 'sample') and (site['is_observed'] is False))
    ]
    samples = mcmc.get_samples()
    num_samples = samples[sample_sites[0]].shape[0]
    unconstrained_samples = unconstrained_vmap(
        model,
        model_args,
        model_kwargs,
        {site: samples[site] for site in sample_sites},
        jnp.arange(num_samples)
    )
    w_mean = {}
    w_sigma = {}
    w1_sum = samples[w1_name].sum()
    for site in sample_sites:
        x = unconstrained_samples[site]
        num_dims = x.ndim
        w1_shape = (-1,) + (1,) * (num_dims - 1)
        w1 = samples[w1_name].reshape(*w1_shape)
        w_mean[site] = (w1 * x).sum(axis=0) / w1_sum
        w_sigma[site] = jnp.sqrt((w1 * (x - w_mean[site])**2).sum(axis=0) / w1_sum)

    new_params = {
        'auto_loc': jnp.hstack([w_mean[key].ravel() for key in sample_sites]),
        'auto_scale': jnp.hstack([w_sigma[key].ravel() for key in sample_sites])
    }
    new_neg_elbo = nlogZ(neg_elbo, samples, w0_name=w0_name, w1_name=w1_name)
    return new_params, new_neg_elbo


@jax.jit
def get_value_from_index(xs, i):
    return jax.tree.map(lambda x: x[i], xs)


@partial(jax.vmap, in_axes=(None, None, None, None, 0))
def unconstrained_vmap(model, model_args, model_kwargs, params, i):
    return numpyro.infer.util.unconstrain_fn(
        model,
        model_args,
        model_kwargs,
        get_value_from_index(params, i)
    )
