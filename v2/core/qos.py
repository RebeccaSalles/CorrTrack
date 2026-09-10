"""P/E cores on macOS through the thread QoS class (Apple Silicon).

macOS does NOT expose CPU affinity (no `taskset` equivalent): a process cannot
be pinned to the P or E cores. It can only be *suggested* through the **QoS**:
a high QoS (user-initiated) is scheduled on the **P-cores**, a low QoS
(background) on the **E-cores**. This is a hint, not a guarantee.

`set_qos("perf"|"eco")` sets the QoS of the current thread; passed as the
`initializer` of a `ProcessPoolExecutor`, it applies to every worker. No-op
outside macOS or when the call fails.
"""

import ctypes
import sys

# values of the qos_class_t enum (<sys/qos.h>)
_QOS = {
    "perf": 0x19,   # QOS_CLASS_USER_INITIATED -> P-cores
    "eco": 0x09,    # QOS_CLASS_BACKGROUND     -> E-cores
}


def set_qos(level):
    """Apply the QoS to the current thread. Return True when applied."""
    cls = _QOS.get(level)
    if cls is None or sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        # int pthread_set_qos_class_self_np(qos_class_t, int relative_priority)
        return libc.pthread_set_qos_class_self_np(ctypes.c_int(cls), ctypes.c_int(0)) == 0
    except Exception:
        return False
