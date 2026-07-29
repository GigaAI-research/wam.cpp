import numpy as np
import wam


def predict_once(model_path, library_path, images, state, instruction):
    with wam.Pipeline.load(
        model_path,
        library=library_path,
        runtime_config=wam.RuntimeConfig(backend="cuda", precision="bf16"),
        session_config=wam.SessionConfig(random_seed=0),
    ) as pipeline:
        return pipeline.predict(images, np.asarray(state, dtype=np.float32),
                                instruction=instruction).action
