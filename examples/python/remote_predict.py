import numpy as np
import wam


def predict_once(host, port, descriptor, environment, images, state, instruction):
    with wam.Client(host, port, descriptor, environment) as client:
        action, stats = client.predict(
            images, np.asarray(state, dtype=np.float32), instruction)
        return action, stats
