import re

# print(re.find(r'ppp', 'pbluered\nflying blue bird')) 


text = "ash is the purest white"
print(re.findall(r'\w+e(?!\w+)', text))

m = re.match(r"(\w+) (\w+)", "Isaac Newton, physicist")
print(m)

m = re.split(r'\W+', 'Mission impossible 7.53=25', 2)
print(m)