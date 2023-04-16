class NeuralNetwork:
    def __init__(self, trainning_data, trainning_labels, _lambda=0.1, bias=0):
        self.weights = len(trainning_data[0])*[0.1]
        self.delta_w = len(trainning_data[0])*[0]

        self.size = len(self.weights)

        self.trainning_data = trainning_data
        self.trainning_labels = trainning_labels

        self.bias = bias
        self._lambda = _lambda

        self.outputs_history = []
        self.evaluated_outputs = []

    @staticmethod
    def phi(z):
        """
        Função de ativação
        :param z:
        :return:
        """
        return 1 if z > 0 else 0

    def neuron(self, data):
        """
        Calcula o valor do neurônio
        :return:
        """
        z = 0
        # Σ (x * w) + bias
        # Somatório de todos os pesos * inputs
        for i in range(self.size):
            x = data[i]
            w = self.weights[i]
            z += x * w
        z += self.bias
        return z

    @staticmethod
    def loss_old(y, _y):
        """
        Calcula o erro
        :param y:
        :param _y:
        :return:
        """

        return (y - _y) ** 2

    def loss(self, output):
        #1/2 * Σ (y - y')²
        loss = 0
        for index, data in enumerate(self.trainning_data):
            y = self.trainning_labels[index]
            _y = output[index]
            loss += (y - _y) ** 2

        return 1/2 * loss

    def update_weights_old(self, data, index, _y):
        # Obter o valor esperado
        y = self.trainning_labels[index]
        # Atualizar os pesos
        for i in range(self.size):
            # w = w + λ * (y - y') * x
            self.delta_w[i] = self._lambda * (y - _y) * data[i]
            self.weights[i] += self.delta_w[i]

        # Atualizar o bias
        self.bias += self._lambda * (y - _y)

    def update_weights(self, output):
        """
        Atualiza os pesos do modelo v2

        :param output:
        :return:
        """
        # wi = wi + λ * Σ (y - y') * xj
        for i in range(len(self.weights)):
            w = 0
            for j in range(len(self.trainning_data)):
                y = self.trainning_labels[j]
                _y = output[j]
                x = self.trainning_data[j]
                w += (y - _y) * x[i]

            self.weights[i] += self._lambda * w

        _bias = 0
        for j in range(len(self.trainning_data)):
            y = self.trainning_labels[j]
            _y = output[j]
            _bias += (y - _y)

        self.bias += self._lambda * _bias


    def train(self, epochs=1):
        """
        Treina o modelo
        :param epochs:
        :return:
        """
        max_accuracy = 0
        for epoch in range(epochs):
            output = []
            # loss = 0
            # Pra cada dado de treino
            for index, data in enumerate(self.trainning_data):
                # Calcular o valor do neurônio
                z = self.neuron(data)
                # Aplicar a função de ativação
                _y = self.phi(z)
                # y = self.trainning_labels[index]
                # loss += self.loss_old(y, _y)
                output.append(_y)


            self.update_weights(output)

            loss = self.loss(output)
            self.outputs_history.append(output.copy())
            size = len(output)
            accuracy = sum([1 if output[i] == self.trainning_labels[i] else 0 for i in range(size)]) / size
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                print(f"Epoch: {epoch+1}")
                print(f"Loss: {loss} - Accuracy: {accuracy}")
                print(f"Bias: {self.bias} - Weights: {self.weights}")
                print("--------------------------------------------------")

    def train_old(self, epochs=1):
        """
        Treina o modelo
        :param epochs:
        :return:
        """
        for epoch in range(epochs):
            output = []
            loss = 0
            # Pra cada dado de treino
            for index, data in enumerate(self.trainning_data):

                # Calcular o valor do neurônio
                z = self.neuron(data)

                # Aplicar a função de ativação
                _y = self.phi(z)

                # Atualizar os pesos
                self.update_weights_old(data, index, _y)

                y = self.trainning_labels[index]

                loss += self.loss_old(y, _y)

                output.append(_y)

            size = len(output)
            accuracy = sum([1 if output[i] == self.trainning_labels[i] else 0 for i in range(size)]) / size
            self.outputs_history.append(output.copy())

            print(f"Epoch: {epoch + 1}")
            print(f"Loss: {loss / 2} - Accuracy: {accuracy}")
            print(f"Bias: {self.bias} - Weights: {self.weights}")
            print("--------------------------------------------------")


    def predict(self, test_data):
        """
        Faz uma predição com o modelo
        :param test_data:
        :return:
        """
        # Pra cada dado de teste
        for index, data in enumerate(test_data):
            # Calcular o valor do neurônio
            z = self.neuron(data)

            # Aplicar a função de ativação
            _y = self.phi(z)
            self.evaluated_outputs.append(_y)

        return self.evaluated_outputs

    def evaluate(self, test_data, test_labels):
        """
        Avalia o modelo com os dados de teste
        :param test_data:
        :param test_labels:
        :return:
        """
        size = len(test_data)
        self.predict(test_data)

        accuracy = sum([1 if self.evaluated_outputs[i] == test_labels[i] else 0 for i in range(size)]) / size

        return accuracy


def create_trainning_data():
    return [
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
    ], [
        1, 1, 0, 0
    ]

if __name__ == '__main__':
    trainning_data, trainning_labels = create_trainning_data()
    model = NeuralNetwork(trainning_data, trainning_labels)
    model.train(epochs=6)

    test_data = [
        [0, 0, 0]
    ]

    y = model.predict(test_data)

    accuracy = model.evaluate(trainning_data, trainning_labels)
    k = 0
