"""Environment-independent wam.cpp model server declarations."""


def load_policy(model_path, runtime_options):
    raise NotImplementedError


def create_server(policy, host, port):
    raise NotImplementedError


def serve(server):
    raise NotImplementedError


def main(argv=None):
    raise NotImplementedError


if __name__ == "__main__":
    main()

