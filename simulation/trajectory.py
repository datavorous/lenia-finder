"""Frame sampling for Lenia trajectories."""

import jax


def final_frame(substrate, rng, params, steps, image_size=64):
    state = substrate.init_state(rng, params)

    def step(state, _):
        return substrate.step_state(rng, state, params), None

    state, _ = jax.lax.scan(step, state, None, length=steps)
    return substrate.render_state(state, params, img_size=image_size)


def _sample(substrate, rng, params, steps, samples, image_size):
    if steps < 1 or samples < 1 or steps % samples:
        raise ValueError("steps must be positive and divisible by samples")
    chunk = steps // samples
    state = substrate.init_state(rng, params)

    def step(state, _):
        return substrate.step_state(rng, state, params), None

    def sample_chunk(state, _):
        # ASAL records the state before each chunk's final update.
        before_last, _ = jax.lax.scan(step, state, None, length=chunk - 1)
        image = substrate.render_state(before_last, params, img_size=image_size)
        after_last = substrate.step_state(rng, before_last, params)
        return after_last, image

    final_state, frames = jax.lax.scan(sample_chunk, state, None, length=samples)
    final_image = substrate.render_state(final_state, params, img_size=image_size)
    return frames, final_image


def sampled_frames(substrate, rng, params, steps, samples, image_size=64):
    return _sample(substrate, rng, params, steps, samples, image_size)[0]


def sampled_frames_with_final(substrate, rng, params, steps, samples, image_size=64):
    return _sample(substrate, rng, params, steps, samples, image_size)
