def fun_a():
    global a
    global b
    a = False
    print(a)


def fun_b():
    global a
    global b
    a = True
    print(a)


def fun_c():
    global a
    a -= 1
    print(a)
