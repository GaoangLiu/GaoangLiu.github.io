
def ext_euclid(a, b):
    old_s, s = 1, 0
    old_t, t = 0, 1
    old_r, r = a, b
    if b == 0:
        return 1, 0, a
    else:
        while r > 0:
            q = old_r // r
            old_r, r = r, old_r % r
            old_s, s = s, old_s - q * s
            old_t, t = t, old_t - q * t
    return old_s, old_t, old_r


def extgcd(a, b, x, y):
    if b == 0:  # Base case
        x, y = 1, 0
        return a 
    
    d = extgcd(b, a % b, y, x)
    y -= (a / b) * x 
    return d

def getxy(a, b):
    if a == 1 and b == 0:
        return 1, 0
    x, y = getxy(b, a % b)
    return y, x - (a // b) * y

print(ext_euclid(5, 7))
print(extgcd(5, 7, 1, 1))
print(getxy(11, 8))
