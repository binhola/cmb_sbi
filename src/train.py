from sbi.inference import NPE

def train_npe(theta_train, x_train, prior, batch_size=100, epochs=100):
    inference = NPE(prior=prior)

    density_estimator = inference.append_simulations(
        theta_train, x_train
    ).train(
        training_batch_size=batch_size,
        max_num_epochs=epochs,
    )

    posterior = inference.build_posterior()

    return posterior, density_estimator