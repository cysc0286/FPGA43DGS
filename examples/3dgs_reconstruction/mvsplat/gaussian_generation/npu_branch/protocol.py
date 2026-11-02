"""Host/worker and native bridge compatibility, checked before device creation."""
PROTOCOL_VERSION = 2
BRIDGE_ABI_VERSION = 2


def check_command(command, sequence, names):
    if command == {"command": "QUIT"}:
        return False
    if (set(command) != {"command", "name", "sequence"} or
            command["command"] != "FORWARD" or
            type(command["sequence"]) is not int or command["sequence"] != sequence or
            command["name"] not in names):
        raise ValueError("Unknown command, stale sequence or unknown partition")
    return True


def check_bridge(library):
    import ctypes as ct
    library.mgs_abi_version.argtypes = []
    library.mgs_abi_version.restype = ct.c_int
    if library.mgs_abi_version() != BRIDGE_ABI_VERSION:
        raise ValueError("Native bridge ABI mismatch; rebuild libmgs_npu.so")


def check_direct_sfb(library):
    """An old bridge must not silently ignore an enabled transfer candidate."""
    import ctypes as ct
    try:
        capability = library.mgs_direct_sfb_supported
    except AttributeError:
        raise ValueError('Bridge lacks direct SFB support; rebuild candidate library') from None
    capability.argtypes = []
    capability.restype = ct.c_int
    if capability() != 1:
        raise ValueError('Bridge rejected direct SFB capability')
