import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Clase para el entorno del juego Tres en Raya
class TicTacToe:
    def __init__(self):
        # Inicializa el tablero como una matriz 3x3 de ceros
        self.board = np.zeros((3,3))
        # El jugador 1 comienza (1 para el jugador, -1 para la IA)
        self.current_player = 1

    def reset(self):
        # Reinicia el tablero y el jugador actual
        self.board = np.zeros((3,3))
        self.current_player = 1
        # Devuelve el estado inicial del juego
        return self.get_state()

    def get_state(self):
        # Devuelve el estado del tablero como un array 1D
        return self.board.flatten()

    def make_move(self, action):
        # Convierte la acción (0-8) en coordenadas del tablero
        row, col = action // 3, action % 3
        # Si la celda está vacía, realiza el movimiento
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            # Cambia al siguiente jugador
            self.current_player = -self.current_player
            return True
        # Si la celda está ocupada, el movimiento es inválido
        return False

    def check_winner(self):
        # Comprueba las filas y columnas
        for i in range(3):
            if abs(np.sum(self.board[i,:])) == 3 or abs(np.sum(self.board[:,i])) == 3:
                return self.board[i,0]
        # Comprueba las diagonales
        if abs(np.trace(self.board)) == 3 or abs(np.trace(np.fliplr(self.board))) == 3:
            return self.board[1,1]
        # Comprueba si hay empate (tablero lleno)
        if np.all(self.board != 0):
            return 0
        # Si no hay ganador ni empate, devuelve None
        return None

# Agente de Q-Learning
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Definimos learning_rate aquí en el constructor
        self.learning_rate = 0.001
        self.model = self.build_model()
        # Parámetros para la exploración epsilon-greedy
        self.epsilon = 1.0  # Probabilidad inicial de exploración
        self.epsilon_min = 0.01  # Probabilidad mínima de exploración
        self.epsilon_decay = 0.995  # Tasa de decaimiento de epsilon
        self.gamma = 0.95  # Factor de descuento para recompensas futuras

    def build_model(self):
        model = keras.Sequential([
        keras.layers.Input(shape=(self.state_size,)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(self.action_size, activation='linear')
    ])
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
                  loss=tf.keras.losses.MeanSquaredError())
        return model

    def act(self, state):
        # Elige una acción usando la política epsilon-greedy
        if np.random.rand() <= self.epsilon:
            # Exploración: elige una acción aleatoria
            return np.random.randint(self.action_size)
        # Explotación: elige la mejor acción según el modelo actual
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        # Actualiza el modelo Q basado en la ecuación de Bellman
        target = reward
        if not done:
            # Si el juego no ha terminado, incluye la estimación de recompensas futuras
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
        target_f = self.model.predict(state.reshape(1, -1))
        target_f[0][action] = target
        # Entrena el modelo para un solo paso
        self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        # Reduce epsilon para disminuir la exploración con el tiempo
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        # Guarda el modelo en un archivo
        self.model.save(filename)

    def load_model(self, filename):
      if os.path.exists(filename):
        self.model = tf.keras.models.load_model(filename, compile=False)
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.MeanSquaredError())
        print(f"Modelo cargado desde {filename}")
      else:
        print(f"No se encontró el archivo {filename}. Se usará un nuevo modelo.")

# Función de entrenamiento
def train_agent(episodes=50, model_file='tictactoe_model.h5'):
    env = TicTacToe()
    agent = QLearningAgent(9, 9)
    
    # Intenta cargar un modelo existente
    if os.path.exists(model_file):
        agent.load_model(model_file)
        print(f"Modelo cargado desde {model_file}. Saltando el entrenamiento.")
        return agent
    
    print("No se encontró un modelo existente. Iniciando entrenamiento...")
    for e in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            if env.make_move(action):
                next_state = env.get_state()
                winner = env.check_winner()
                if winner is not None:
                    done = True
                    reward = 1 if winner == 1 else -1 if winner == -1 else 0
                else:
                    reward = 0
                agent.train(state, action, reward, next_state, done)
                state = next_state
        
        if e % 1000 == 0:
            print(f"Episodio: {e}/{episodes}")
    
    # Guarda el modelo entrenado
    agent.save_model(model_file)
    print(f"Modelo guardado en {model_file}")
    return agent


# Función para jugar contra el agente entrenado
def play_against_agent(agent):
    env = TicTacToe()
    state = env.reset()
    done = False
    while not done:
        print(env.board)
        if env.current_player == 1:
            while True:
                try:
                       action = int(input("Tu turno (0-8): "))
                    if 0 <= action <= 8:
                        break
                    else:
                        print("Por favor, introduce un número entre 0 y 8.")
                except ValueError:
                    print("Por favor, introduce un número válido.")
        else:
            action = agent.act(state)
        if env.make_move(action):
            state = env.get_state()
            winner = env.check_winner()
            if winner is not None:
                done = True
                print(env.board)
                if winner == 0:
                    print("Empate!")
                else:
                    print(f"Ganador: {'Tú' if winner == 1 else 'IA'}")
        else:
            print("Movimiento inválido, intenta de nuevo.")


# Función principal
def main():
    model_file = 'tictactoe_model.h5'
    trained_agent = train_agent(model_file=model_file)
    
    while True:
        play_against_agent(trained_agent)
        play_again = input("¿Quieres jugar otra vez? (s/n): ").lower()
        if play_again != 's':
            break

    print("¡Gracias por jugar!")

if __name__ == "__main__":
    main()
