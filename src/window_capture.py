"""
Window Capture Module for Nikke Math Solver

Provides functionality to detect, lock onto, and capture the nikke.exe game window.
Uses Windows API via pywin32 for window management and mss for screen capture.
"""

import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, List
from dataclasses import dataclass

import psutil
import mss
from PIL import Image


# Windows API constants
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
GW_HWNDNEXT = 2

# PrintWindow constants
PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 2

# GDI constants
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0


@dataclass
class WindowInfo:
    """Information about a detected window."""
    hwnd: int
    pid: int
    title: str
    rect: Tuple[int, int, int, int]  # (x, y, width, height)


def is_nikke_running(process_name: str = "nikke.exe") -> bool:
    """
    Check if the target process is currently running.

    Args:
        process_name: Name of the process to find (default: nikke.exe)

    Returns:
        True if process is running, False otherwise

    Example:
        >>> is_nikke_running()
        True
        >>> is_nikke_running("notepad.exe")
        False
    """
    process_name_lower = process_name.lower()
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'].lower() == process_name_lower:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


def get_process_ids(process_name: str = "nikke.exe") -> List[int]:
    """
    Get all process IDs for a given process name.

    Args:
        process_name: Name of the process to find

    Returns:
        List of process IDs matching the name

    Example:
        >>> get_process_ids("nikke.exe")
        [12345]
    """
    pids = []
    process_name_lower = process_name.lower()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'].lower() == process_name_lower:
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return pids


def _enum_windows_callback(hwnd: int, results: list) -> bool:
    """Callback for EnumWindows - collects visible windows."""
    if ctypes.windll.user32.IsWindowVisible(hwnd):
        results.append(hwnd)
    return True


def _get_window_thread_process_id(hwnd: int) -> int:
    """Get the process ID associated with a window handle."""
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _get_window_text(hwnd: int) -> str:
    """Get the title text of a window."""
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
    buffer = ctypes.create_unicode_buffer(length)
    ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length)
    return buffer.value


def _get_window_rect_raw(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the bounding rectangle of a window.

    Returns:
        Tuple of (left, top, right, bottom) or None if failed
    """
    rect = wintypes.RECT()
    if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return (rect.left, rect.top, rect.right, rect.bottom)
    return None


def find_nikke_window(process_name: str = "nikke.exe") -> Optional[WindowInfo]:
    """
    Find the main window handle for the target process.

    Searches for visible windows belonging to the specified process and returns
    information about the first match found.

    Args:
        process_name: Name of the process to find (default: nikke.exe)

    Returns:
        WindowInfo object if found, None otherwise

    Example:
        >>> info = find_nikke_window()
        >>> if info:
        ...     print(f"Found window: {info.title} at {info.rect}")
    """
    # Get all PIDs for the target process
    pids = get_process_ids(process_name)
    if not pids:
        return None

    # Enumerate all visible windows
    windows = []
    enum_callback = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        wintypes.HWND,
        wintypes.LPARAM
    )

    @enum_callback
    def callback(hwnd, lparam):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            windows.append(hwnd)
        return True

    ctypes.windll.user32.EnumWindows(callback, 0)

    # Find windows belonging to our target process
    for hwnd in windows:
        window_pid = _get_window_thread_process_id(hwnd)
        if window_pid in pids:
            rect_raw = _get_window_rect_raw(hwnd)
            if rect_raw:
                left, top, right, bottom = rect_raw
                # Filter out windows with zero or negative size
                width = right - left
                height = bottom - top
                if width > 0 and height > 0:
                    return WindowInfo(
                        hwnd=hwnd,
                        pid=window_pid,
                        title=_get_window_text(hwnd),
                        rect=(left, top, width, height)
                    )

    return None


def get_window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the current position and size of a window.

    Args:
        hwnd: Window handle

    Returns:
        Tuple of (x, y, width, height) or None if window not found

    Example:
        >>> rect = get_window_rect(hwnd)
        >>> if rect:
        ...     x, y, width, height = rect
        ...     print(f"Window at ({x}, {y}) size {width}x{height}")
    """
    rect_raw = _get_window_rect_raw(hwnd)
    if rect_raw:
        left, top, right, bottom = rect_raw
        return (left, top, right - left, bottom - top)
    return None


def is_window_valid(hwnd: int) -> bool:
    """
    Check if a window handle is still valid.

    Args:
        hwnd: Window handle to check

    Returns:
        True if window exists and is valid, False otherwise
    """
    return bool(ctypes.windll.user32.IsWindow(hwnd))


def is_window_visible(hwnd: int) -> bool:
    """
    Check if a window is visible (not minimized or hidden).

    Args:
        hwnd: Window handle to check

    Returns:
        True if window is visible, False otherwise
    """
    return bool(ctypes.windll.user32.IsWindowVisible(hwnd))


def is_window_minimized(hwnd: int) -> bool:
    """
    Check if a window is minimized.

    Args:
        hwnd: Window handle to check

    Returns:
        True if window is minimized, False otherwise
    """
    return bool(ctypes.windll.user32.IsIconic(hwnd))


def _capture_with_printwindow(hwnd: int, width: int, height: int) -> Optional[Image.Image]:
    """
    Capture window using PrintWindow API (excludes overlays).

    Args:
        hwnd: Window handle
        width: Window width
        height: Window height

    Returns:
        PIL Image or None if failed
    """
    # Get GDI32 and User32
    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    # Get window DC
    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None

    try:
        # Create compatible DC
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        if not mem_dc:
            return None

        try:
            # Create compatible bitmap
            bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
            if not bitmap:
                return None

            try:
                # Select bitmap into memory DC
                old_bitmap = gdi32.SelectObject(mem_dc, bitmap)

                # Use PrintWindow to render window content
                # PW_RENDERFULLCONTENT (2) works better for DWM-composed windows
                result = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)

                if not result:
                    # Try without flags as fallback
                    result = user32.PrintWindow(hwnd, mem_dc, 0)

                if not result:
                    gdi32.SelectObject(mem_dc, old_bitmap)
                    return None

                # Prepare BITMAPINFOHEADER structure
                class BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ('biSize', wintypes.DWORD),
                        ('biWidth', wintypes.LONG),
                        ('biHeight', wintypes.LONG),
                        ('biPlanes', wintypes.WORD),
                        ('biBitCount', wintypes.WORD),
                        ('biCompression', wintypes.DWORD),
                        ('biSizeImage', wintypes.DWORD),
                        ('biXPelsPerMeter', wintypes.LONG),
                        ('biYPelsPerMeter', wintypes.LONG),
                        ('biClrUsed', wintypes.DWORD),
                        ('biClrImportant', wintypes.DWORD),
                    ]

                bmi = BITMAPINFOHEADER()
                bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.biWidth = width
                bmi.biHeight = -height  # Negative for top-down DIB
                bmi.biPlanes = 1
                bmi.biBitCount = 32
                bmi.biCompression = BI_RGB
                bmi.biSizeImage = width * height * 4

                # Create buffer for pixel data
                buffer = ctypes.create_string_buffer(width * height * 4)

                # Get bitmap bits
                lines = gdi32.GetDIBits(
                    mem_dc, bitmap, 0, height,
                    buffer, ctypes.byref(bmi), DIB_RGB_COLORS
                )

                # Restore old bitmap
                gdi32.SelectObject(mem_dc, old_bitmap)

                if lines == 0:
                    return None

                # Convert BGRA to RGB
                raw_data = buffer.raw
                img = Image.frombuffer('RGBA', (width, height), raw_data, 'raw', 'BGRA', 0, 1)
                return img.convert('RGB')

            finally:
                gdi32.DeleteObject(bitmap)
        finally:
            gdi32.DeleteDC(mem_dc)
    finally:
        user32.ReleaseDC(hwnd, hwnd_dc)


def _capture_with_mss(x: int, y: int, width: int, height: int) -> Optional[Image.Image]:
    """
    Capture screen region using mss (fallback method).

    Note: This captures screen pixels including any overlays.

    Args:
        x: Left position
        y: Top position
        width: Width
        height: Height

    Returns:
        PIL Image or None if failed
    """
    try:
        with mss.mss() as sct:
            monitor = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }
            screenshot = sct.grab(monitor)
            img = Image.frombytes(
                "RGB",
                (screenshot.width, screenshot.height),
                screenshot.rgb
            )
            return img
    except Exception:
        return None


def capture_window(hwnd: int) -> Optional[Image.Image]:
    """
    Capture the contents of a window as a PIL Image.

    Uses PrintWindow API to capture window content directly, which excludes
    overlay windows. Falls back to mss screen capture if PrintWindow fails.

    Args:
        hwnd: Window handle to capture

    Returns:
        PIL Image of window contents, or None if capture failed

    Example:
        >>> img = capture_window(hwnd)
        >>> if img:
        ...     img.save("screenshot.png")
    """
    # Validate window state
    if not is_window_valid(hwnd):
        return None

    if is_window_minimized(hwnd):
        return None

    # Get window position and size
    rect = get_window_rect(hwnd)
    if not rect:
        return None

    x, y, width, height = rect

    # Validate dimensions
    if width <= 0 or height <= 0:
        return None

    # Try PrintWindow first (excludes overlays)
    img = _capture_with_printwindow(hwnd, width, height)
    if img is not None:
        return img

    # Fallback to mss (includes overlays but better than nothing)
    return _capture_with_mss(x, y, width, height)


class WindowCapture:
    """
    Manages continuous window capture and monitoring.

    Provides a higher-level interface for tracking a game window,
    detecting when it moves or closes, and capturing frames.

    Example:
        >>> capture = WindowCapture()
        >>> if capture.find_window():
        ...     print(f"Found: {capture.window_info.title}")
        ...     while capture.is_active():
        ...         frame = capture.grab_frame()
        ...         if frame:
        ...             process_frame(frame)
    """

    def __init__(self, process_name: str = "nikke.exe"):
        """
        Initialize WindowCapture.

        Args:
            process_name: Name of the target process (default: nikke.exe)
        """
        self.process_name = process_name
        self.window_info: Optional[WindowInfo] = None
        self._last_rect: Optional[Tuple[int, int, int, int]] = None

    def find_window(self) -> bool:
        """
        Attempt to find and lock onto the target window.

        Returns:
            True if window was found, False otherwise
        """
        self.window_info = find_nikke_window(self.process_name)
        if self.window_info:
            self._last_rect = self.window_info.rect
            return True
        return False

    def is_active(self) -> bool:
        """
        Check if the tracked window is still valid and visible.

        Returns:
            True if window is active, False if lost or minimized
        """
        if not self.window_info:
            return False

        hwnd = self.window_info.hwnd
        return is_window_valid(hwnd) and not is_window_minimized(hwnd)

    def has_moved(self) -> bool:
        """
        Check if the window has moved or resized since last check.

        Returns:
            True if window position/size changed, False otherwise
        """
        if not self.window_info:
            return False

        current_rect = get_window_rect(self.window_info.hwnd)
        if current_rect != self._last_rect:
            self._last_rect = current_rect
            # Update stored info
            if current_rect:
                self.window_info.rect = current_rect
            return True
        return False

    def get_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get current window rectangle.

        Returns:
            Tuple of (x, y, width, height) or None
        """
        if not self.window_info:
            return None
        return get_window_rect(self.window_info.hwnd)

    def grab_frame(self) -> Optional[Image.Image]:
        """
        Capture the current window contents.

        Returns:
            PIL Image of window, or None if capture failed
        """
        if not self.window_info:
            return None
        return capture_window(self.window_info.hwnd)

    def release(self):
        """Release the tracked window."""
        self.window_info = None
        self._last_rect = None

    def get_status_string(self) -> str:
        """
        Get a human-readable status string for UI display.

        Returns:
            Status string like "nikke.exe (1920x1080)" or "Not detected"
        """
        if not self.window_info:
            return "Not detected"

        if not self.is_active():
            return "Window lost"

        rect = self.get_rect()
        if rect:
            _, _, width, height = rect
            return f"{self.process_name} ({width}x{height})"

        return f"{self.process_name} (unknown size)"
