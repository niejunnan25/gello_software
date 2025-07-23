import pygame
import sys


NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# 按下 ESC 退出
# 按下 s 停止当前的录制，然后 多线程写入图像数据
# 按下空格开始，图形化窗口变成绿色
KEY_ESC_RECORDING = pygame.K_ESCAPE
KET_STOP_RECORDING = pygame.K_s
KEY_START_RECORDING = pygame.K_SPACE

class KBReset:

    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 800))
        self._set_color(NORMAL)
        self._saved = False

    def update(self) -> str:
        pressed_list = self._get_pressed()

        if KEY_ESC_RECORDING in pressed_list:
            return "esc"

        if KET_STOP_RECORDING in pressed_list:
            self._set_color(RED)
            self._saved = False
            return "stop"

        if self._saved:
            return "save"

        if KEY_START_RECORDING in pressed_list:
            self._set_color(GREEN)
            self._saved = True
            return "start"

        self._set_color(NORMAL)
        return "normal"

    def _get_pressed(self):
        pressed = []
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed

    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()

    def close(self):
        pygame.display.quit()
        pygame.quit()

def main():
    kb = KBReset()

    flag = True
    while flag:
        state = kb.update()
        if state == "esc":
            flag = False
    # print(kb._get_pressed())
    kb.close()

if __name__ == "__main__":
    main()