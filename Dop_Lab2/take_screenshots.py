"""Скриншоты работающих приложений (серверы streamlit должны быть запущены на портах 8601 и 8602)."""
from playwright.sync_api import sync_playwright

def settle(pg, ms=2500):
    pg.wait_for_timeout(ms)

def shot(pg, name):
    pg.screenshot(path=f"screenshots/{name}.png", full_page=True)

def press_slider(pg, label_part, key, times):
    """Сдвинуть слайдер, чей подпись содержит label_part, клавишей key."""
    s = pg.locator('[data-testid="stSlider"]', has_text=label_part).locator('input[type="range"]').first
    s.focus()
    for _ in range(times):
        pg.keyboard.press(key)
    settle(pg, 800)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 2300}, device_scale_factor=1.5)

    # ---------- Вариант 1 ----------
    pg.goto("http://localhost:8601"); settle(pg, 4000)
    shot(pg, "v1_01_start")
    pg.get_by_role("button", name="Обучить модель").click(); settle(pg, 4000)
    shot(pg, "v1_02_default")
    press_slider(pg, "n_estimators", "ArrowRight", 20)      # 100 -> 300 деревьев
    press_slider(pg, "max_depth", "ArrowRight", 3)          # 5 -> 8
    press_slider(pg, "Доля тестовой", "ArrowRight", 1)      # 0.25 -> 0.30
    pg.get_by_role("button", name="Обучить модель").click(); settle(pg, 5000)
    shot(pg, "v1_03_tuned")

    # ---------- Вариант 2 ----------
    pg.goto("http://localhost:8602"); settle(pg, 4000)
    shot(pg, "v2_01_start")
    pg.get_by_role("button", name="Обучить выбранные модели").click(); settle(pg, 4000)
    shot(pg, "v2_02_two_models")
    box = pg.locator('[data-testid="stMultiSelect"]')
    for name in ["k ближайших соседей", "SVM (RBF)", "Дерево решений", "Градиентный бустинг"]:
        box.locator("input").click(); pg.keyboard.type(name); pg.keyboard.press("Enter"); pg.keyboard.press("Escape"); settle(pg, 700)
    pg.get_by_role("button", name="Обучить выбранные модели").click(); settle(pg, 7000)
    shot(pg, "v2_03_all_models")
    b.close()
print("done")

# обрезаем пустое поле внизу страницы
from PIL import Image, ImageChops
import glob
for f in glob.glob("screenshots/v*.png"):
    im = Image.open(f).convert("RGB")
    main = im.crop((520, 0, im.width, im.height))          # основная область, без боковой панели
    bg = Image.new("RGB", main.size, main.getpixel((main.width - 5, main.height - 5)))
    bbox = ImageChops.difference(main, bg).getbbox()
    if bbox:
        im.crop((0, 0, im.width, min(im.height, bbox[3] + 60))).save(f)
