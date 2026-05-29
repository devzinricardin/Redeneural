import numpy as np

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([2,4,6,8,10,12])


class RedeLinear:
    def __init__(self):
        self.w = np.random.uniform(0, 3)
        self.b = 0.5
        self.eta = 0.003
    def ativacao(self, x):
        return x
    def saida(self, x):
        cal = x * self.w + self.b
        return self.ativacao(cal)
    def treino(self, X, y, epocas=2000):
        for epoca in range(epocas):
            for x,target in zip(X, y):
                prev = self.saida(x)
                erro = target - prev

                self.w += self.eta * x * erro
                self.b += self.eta * erro

modelo = RedeLinear()
modelo.treino(x_train, y_train)

X = np.array([5,6,7,8])
y = np.array([10,12,14,16])
for x in X:
    print(x, f"Saida:  {modelo.saida(x):.1f}")
print()
print("Correto: ")
for yi in y:
    print(yi)
