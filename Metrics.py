import numpy as np

class Metrics:
    def __init__(self, lvl):
        self.lvl = lvl
    
    def P(self, marks):
        x = np.array(marks) > 2
        return float(sum(x) / self.lvl)
    
    def CG(self, marks):
        return sum(marks[:self.lvl])
    
    def DCG(self, marks):
        x = np.array(marks[:self.lvl])
        y = np.log2(np.arange(2, self.lvl + 2))
        return sum(x / y)
    
    def NDCG(self, marks):
        IDCG = 5 * self.lvl
        return self.DCG(marks) / IDCG
    
    def ERR(self, marks):
        mx = max(marks[:self.lvl])
        grades = (np.power(2, np.array(marks[:self.lvl]) - 1) / np.power(2, mx))
        err = 0
        p = 1
        r = 1
        for grade in grades:
            err += p * grade / r
            p *= (1 - grade)
            r += 1
        return err

    def measure(self, marks):
        return list(map(lambda f: f(marks), [self.P, self.CG, self.DCG, self.NDCG, self.ERR]))


if __name__ == '__main__':
    marks = [4, 4, 3, 1, 3]
    m1 = Metrics(1)
    m3 = Metrics(3)
    m5 = Metrics(5)
    print('@1 ', m1.measure(marks[:1]))
    print('@3 ', m3.measure(marks[:3]))
    print('@5 ', m5.measure(marks[:5]))
