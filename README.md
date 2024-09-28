# Modelo Perceptron em Python

Este repositório contém um exemplo de implementação de um modelo Perceptron em Python usando a biblioteca `scikit-learn`. O Perceptron é um algoritmo de aprendizado supervisionado usado para classificação binária.

## Descrição

O Perceptron é um dos algoritmos mais simples e antigos de aprendizado de máquina. Ele é um classificador linear que tenta encontrar um hiperplano que separa duas classes de dados. Este exemplo demonstra como criar, treinar e avaliar um modelo Perceptron usando um conjunto de dados gerado artificialmente.

## Requisitos

- Python 3.6 ou superior
- `numpy`
- `scikit-learn`

Você pode instalar as bibliotecas necessárias usando o seguinte comando:

```sh
pip install numpy scikit-learn
```

## Uso

### 1. Gerar um conjunto de dados

O código gera um conjunto de dados de exemplo com 1000 amostras e 10 características. O conjunto de dados é dividido em conjuntos de treinamento e teste.

### 2. Criar e treinar o modelo Perceptron

O modelo Perceptron é criado e treinado usando os dados de treinamento.

### 3. Fazer previsões e avaliar o modelo

O modelo treinado é usado para fazer previsões com os dados de teste, e a precisão do modelo é calculada.

## Código

Aqui está o código completo:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gerar um conjunto de dados de exemplo
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo Perceptron
model = Perceptron()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular a precisão
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')
```

## Executando o Código

Para executar o código, siga estas etapas:

1. Clone este repositório:

   ```sh
   git clone https://github.com/alexmontanha/perceptron_sample.git
   ```

2. Navegue até o diretório do repositório:

   ```sh
   cd nome-do-repositorio
   ```

3. Instale as dependências:

   ```sh
   pip install numpy scikit-learn
   ```

4. Execute o script:

   ```sh
   python perceptron_example.py
   ```

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorar este exemplo.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
