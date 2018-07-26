import decimal as dc

# dc.getcontext().rounding = dc.ROUND_HALF_UP
# def call_test(c):
#     print("test before add: %d" % id(c))
#     c += dc.Decimal(0.05)
#     print("test after add: %d" % id(c))
#
#
# if __name__ == "__main__":
#     a = dc.Decimal(1.65)
#     print("main before call test: %d" % id(a))
#     call_test(a)
#     print("main after call test: %d" % id(a))

class AddObj:

    def __init__(self):
        self.a = 2

    def add(self, c):
        print("invoke before: %d" % id(self.a))
        a = self.a + c
        print("invoke after: %d" % id(a))

if __name__ == '__main__':
    class1 = AddObj()
    class1.add(2)
    # AddObj.add(2)