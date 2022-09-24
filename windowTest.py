from win32gui import GetWindowText, GetForegroundWindow, FindWindow, MoveWindow, GetWindowRect, SetForegroundWindow
GAME_NAME = "Jump King"

try:
    gameWin = FindWindow(None, GAME_NAME)
    rect = GetWindowRect(gameWin)
    print(str(rect))
    MoveWindow(gameWin, -8, 0, 967 + 8, 761, False)
    rect = GetWindowRect(gameWin)
    print(str(rect))

    SetForegroundWindow(gameWin)
except Exception:
    print("Can't find game window!")