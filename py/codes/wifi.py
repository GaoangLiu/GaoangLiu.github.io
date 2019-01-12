from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import pickle, time, sys, os, re, random
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup


REGPAGE = 'http://192.168.1.1'

class Wifi():
	def __init__(self):
		options = Options()
		for arg in (
			# '--headless',
			'--disable-gpu',
			'--no-sandbox',
			'disable-infobars'):
			options.add_argument(arg)
		self.driver = webdriver.Chrome(options=options)
		self.driver.maximize_window()
		# self.driver.set_page_load_timeout(30)

	def waitfor(self, label, symbol):
		if label == 'class':
			WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, symbol)))
		elif label == 'id':
			WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, symbol)))

	def login(self):
		''': webdrive object
		'''
		self.driver.get(REGPAGE)
		time.sleep(3)
		print(self.driver.page_source)
		return 
		self.waitfor('class', 'lgPwd')
		self.driver.find_element_by_id('pwdTipStr').click()
		self.driver.find_element_by_id('lgPwd').send_keys('tplink123')
		self.driver.find_element_by_id('loginSub').send_keys(Keys.RETURN)
		# print(self.driver.page_source)
		return 'bMdelTitle' in self.driver.page_source

	def brute_force_login(self):
		self.driver.get(REGPAGE)
		self.waitfor('class', 'lgPwd')
		rockyou = open('rockyou10000.txt', 'r').read().split("\n")
		for cand in rockyou:
			time.sleep(2)
			self.driver.find_element_by_id('lgPwd').clear()
			self.driver.find_element_by_id('pwdTipStr').click()		
			self.driver.find_element_by_id('lgPwd').send_keys(cand)
			self.driver.find_element_by_id('loginSub').send_keys(Keys.RETURN)
		time.sleep(10)



	def rebot(self):
		self.login()
		self.waitfor('id', 'routerSetMbtn')
		self.driver.find_element_by_partial_link_text('1870').click()

		self.waitfor('class','menuLbl')
		self.driver.find_element_by_xpath("//label[@class='menuLbl' and text()='备份和载入配置']").click()

		time.sleep(0.5)
		self.driver.find_element_by_xpath("//input[@type='file']").send_keys('/usr/local/info/config.bin')
		self.driver.find_element_by_id("sysRestore").click()
		self.driver.find_elements_by_class_name("subBtn")[0].click()		
		time.sleep(10)		


	def get_devices(self):
		self.login()
		self.waitfor('id', 'routeMgtMbtn')
		self.driver.find_element_by_id("routeMgtMbtn").click()	
		self.waitfor('class', 'eptConC')
		soup = BeautifulSoup(self.driver.page_source, 'lxml')
		print(soup.findAll('span', {'class':'name'}))






if __name__ == '__main__':
	wifi = Wifi()
	# wifi.brute_force_login()
	# wifi.rebot()		
	wifi.login()
	wifi.driver.quit()




