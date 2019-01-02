import requests

r = requests.get('https://api.github.com/user', auth=('ssrzz', 'github@2018'))
print(r.status_code, r.text, r.json)

s = requests.Session()
print(s.get("http://www.douban.com").text)
