def drop_n_and_freeze(n, model):
    for _ in range(n):
        model.layers.pop()

    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]

    for layer in model.layers:
        layer.trainable = False

    return model
