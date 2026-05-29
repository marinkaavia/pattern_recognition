"""Вспомогательные функции"""

from random import randint

def generate_colors(n):
    return [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(n)]
