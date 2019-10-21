---
layout:     post
title:      Zeller's Congruence
date:       2019-09-09
img:        date.jpg
tags: [algorithm, date]
---

**Zeller's congruence** is an algorithm devised by [Christian Zeller](https://en.wikipedia.org/wiki/Christian_Zeller) to calculate the day of the week for any date. 

The formula is simple (not that simple to remember):
$$d = (D + \lfloor \frac{13(M+1)}{5} \rfloor + Y + \lfloor \frac{Y}{4} \rfloor + \lfloor \frac{C}{4} \rfloor -2C ) \% 7$$ 

where, 
* $$D$$ is the day of the month 
* $$M$$ is the month
    * 3 = March,  4 = April, 12 = December
    * 13 = January, 14 = February of previous year, hence year = year - 1
* $$Y$$ is the year of the century (e.g., 2022 % 100 = 22)
* $$C$$ is the zero-based century ( e.g., $$\lfloor 2022 / 100 \rfloor  = 20$$)

E.g., $$d(2019/09/09) = (9 + 13 * 10 / 5 + 19 + 4 + 5 - 40) \% 7 = 2 $$, means it is Monday

Python Code to get the day. 
```python
class MyDate:
    def dayOfTheWeek(self, d, m, y):
        if m < 3:
            m += 12 
            y -= 1
        c, y = divmod(y, 100)
        res = (d + (13 * (m + 1)) // 5  + y + y // 4 + c // 4 - 2 * c) % 7 
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return days[(res + 6) % 7]
```

The result $$d$$ is the zero-based day of the week. It starts from Saturday(0 = Saturday), and ends on Friday (6 = Friday). For an ISO week date Day-of-Week d (1 = Monday to 7 = Sunday), a variant of formula is:

$$d = ((d + 5) \% 7) + 1 $$



