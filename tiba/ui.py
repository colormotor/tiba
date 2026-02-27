#!/usr/bin/env python3
import imgui
import numpy as np

defaults = lambda: None
defaults.highlight = [0.8, 0.5, 0.0, 1.0]

# GUI helpers
def imgui_title(text):
    imgui.push_style_color(imgui.COLOR_HEADER, 0.3, 0.3, 0.3, 1.0)
    imgui.selectable(text, True)
    imgui.pop_style_color()

def imgui_info_text(text):
    imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
    imgui.text(text)
    imgui.pop_style_color()

def imgui_text_separator(text):
    if not imgui.get_current_context():
        return
    imgui.separator()
    imgui.text(text)
    imgui.separator()


def imgui_color_button(text, color=defaults.highlight):
    if len(color) == 3:
        color = np.concatenate([color, [1.0]])
    imgui.push_style_color(imgui.COLOR_BUTTON, *color)
    res = imgui.button(text)
    imgui.pop_style_color()
    return res

def imgui_color_checkbox(text, value, color=defaults.highlight):
    if len(color) == 3:
        color = np.concatenate([color, [1.0]])
    imgui.push_style_color(imgui.COLOR_BUTTON, *color)
    changed, value = imgui.checkbox(text, value)
    imgui.pop_style_color()
    return changed, value
