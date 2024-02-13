import pkg_resources

TORCH_VERSION = pkg_resources.get_distribution("torch").version
CPU_ONLY_TORCH_URL = "https://download.pytorch.org/whl/cpu/"
