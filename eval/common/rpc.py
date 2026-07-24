"""Shared RPC and PolicySpec wire declarations for evaluation clients."""


def encode_tensor(tensor):
    raise NotImplementedError


def decode_tensor(message):
    raise NotImplementedError


def decode_policy_spec(message):
    raise NotImplementedError


class RpcClient:
    def get_model_info(self):
        raise NotImplementedError

    def predict(self, request):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

