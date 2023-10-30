from time import sleep

import undetected_chromedriver as uc

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def main():
    driver = uc.Chrome(use_subprocess=True)
    driver.get("http://192.168.0.121:4455/")
    driver.find_element(By.CSS_SELECTOR, ".genOptions").click()
    elem = WebDriverWait(driver=driver, timeout=20.0).until(
        EC.element_to_be_clickable((By.ID, "idea-important"))
    )
    elem.click()
    while True:
        try:
            elem = driver.find_element(By.CSS_SELECTOR, ".ttm-input")
            input_value = elem.get_dom_attribute("value")
            if not input_value:
                break
        except ConnectionError:
            pass
    sleep(1)
    elem = driver.find_element(By.CSS_SELECTOR, ".ttm-send-btn")
    elem.click()

    sleep(100000)


if __name__ == '__main__':
    main()
