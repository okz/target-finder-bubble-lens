"""Window helpers shared by Qt overlays."""

from __future__ import annotations

import sys
import traceback
from ctypes import c_void_p

_qt_crash_guard_installed = False


def install_qt_crash_guard() -> None:
    """Stop PyQt6 from calling ``abort()`` on an unhandled slot exception.

    Without a custom ``sys.excepthook``, an exception escaping a Qt slot
    (e.g. a QTimer callback) makes PyQt6 call ``qFatal()`` and abort the
    process. Logging the exception here instead keeps the app running.
    """
    global _qt_crash_guard_installed
    if _qt_crash_guard_installed:
        return
    _qt_crash_guard_installed = True

    def _hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def warm_up_macos_keyboard_layout() -> None:
    """Pre-fetch the current keyboard input source on the main thread.

    pynput's macOS keyboard.Listener resolves the keyboard layout on its own
    background thread the first time it starts, which can trip a
    dispatch_assert_queue check and crash the process. Doing the same lookup
    here first, on the main thread, avoids that.
    """
    if sys.platform != "darwin":
        return
    try:
        from pynput._util.darwin import keycode_context

        with keycode_context():
            pass
    except Exception:
        pass


def raise_macos_window_above_system_ui(widget, *, level_offset: int = 0) -> bool:
    """Raise a Qt widget above the macOS menu bar/Dock without using Spaces fullscreen."""
    if sys.platform != "darwin":
        return False
    try:
        import AppKit
        import objc

        widget.winId()
        native_obj = objc.objc_object(c_void_p=c_void_p(int(widget.winId())))
        ns_window = native_obj.window() if hasattr(native_obj, "window") else native_obj
        if ns_window is None or not hasattr(ns_window, "setLevel_"):
            return False

        level = int(AppKit.NSStatusWindowLevel) + int(level_offset)
        ns_window.setLevel_(level)
        behavior = (
            int(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces)
            | int(AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
        )
        if hasattr(ns_window, "setCollectionBehavior_"):
            ns_window.setCollectionBehavior_(behavior)
        return True
    except Exception:
        return False
