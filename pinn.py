import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#from scipy.interpolate import griddata
#from plotting import newfig, savefig
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

####Solving diffusion equation#####

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10.0        # Length of the rod
T = 1.0         # Total time
D = 1.0         # Diffusion coefficient
N_x = 100       # Number of spatial points
N_t = 1000      # Number of time steps
dx = L / N_x    # Spatial step size
dt = T / N_t    # Time step size

# Stability condition for explicit scheme
alpha = D * dt / dx**2
if alpha > 0.5:
    print(f"Warning: Stability condition not met (alpha = {alpha:.2f} > 0.5)")

# Initial and boundary conditions
u = np.zeros(N_x)
u[int(N_x / 4):int(3 * N_x / 4)] = 1.0  # Initial condition: a square pulse

u1 = np.copy(u)

print(u1)
print(u)

u2 = np.zeros(N_x)
u2[int(N_x / 4):int(3 * N_x / 4)] = 1.0  # Initial condition: a square pulse

# Pre-allocate the solution array
u_new = np.zeros(N_x)
u_star = []

# Time-stepping loop
for n in range(N_t):
    for i in range(1, N_x - 1):
        u_new[i] = u[i] + alpha * (u[i + 1] - 2 * u[i] + u[i - 1])

    # Update u
    #u[:] = u_new[:]
    u = np.copy(u_new)

    # Apply boundary conditions (Dirichlet)
    u[0] = 0.0
    u[-1] = 0.0
    #print(u)
    u_star.append(u)

#print(u1)
#print(u)

u_star = np.array(u_star)
print(u_star)

# Plot the results
x = np.linspace(0, L, N_x)
t = np.linspace(0, T, N_t)
plt.plot(x, u1, label=f'Time = {T}')
plt.plot(x, u, label=f'Time = {T}')
plt.xlabel('Position (x)')
plt.ylabel('u(x, t)')
plt.title('1D Diffusion Equation')
plt.legend()
plt.show()


##############################################

u_star = u_star.reshape(100000, 1)


X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

####Initial Condiditon

X_ic = X_star[0:100]
u_ic = np.zeros(N_x)
u_ic[int(N_x / 4):int(3 * N_x / 4)] = 1.0  # Initial condition: a square pulse

### Boundary Condition

X_bc1 = X_star[::100]
u_bc1 = np.zeros((1000, 1))

X_bc2 = X_star[99::100]
u_bc2 = np.zeros((1000, 1))


## One set of train test data
x_train,x_test,y_train,y_test = train_test_split(X_star,u_star,test_size=0.45,random_state=4)


## Another set of train test data
N_u = 6000
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx,:]
u_train = u_star[idx,:]
print(idx)


## The MLP mpdel

def build_model():

  model = Sequential()
  model.add(Dense(30, activation='tanh', input_dim = x_train.shape[1]))
  from keras.layers import Dropout
  model.add(Dense(30, activation = 'tanh'))
  model.add(Dropout(0.2))
  model.add(Dense(30, activation='tanh'))
  model.add(Dense(30, activation='tanh'))
  model.add(Dense(30, activation='tanh'))
  model.add(Dense(30, activation='tanh'))
  #model.add(Dense(20, activation='relu'))
  #model.add(Dense(20, activation='relu'))
  #model.add(Dense(64, activation='relu'))
  model.add(Dense(1))

  return model

model = build_model()


## The residual calcualtion and loss functions

def compute_pde_residual(model, X):

    lambda_2 = tf.exp(-6.0)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        u = model(X)
        u_x = tape.gradient(u, X)[:, 0]
        u_t = tape.gradient(u, X)[:, 1]

        u_xx = tape.gradient(u_x, X)[:, 0]

    pde_residual = u_t - u_xx
    #pde_residual = u_t + lambda_1*u*u_x - lambda_2*u_xx
    return pde_residual

def boundary_condition_loss1(model, X_bc1, u_bc1):
    u_pred = model(X_bc1)
    return tf.reduce_mean(tf.square(u_pred - u_bc1))


def boundary_condition_loss2(model, X_bc2, u_bc2):
    u_pred = model(X_bc2)
    return tf.reduce_mean(tf.square(u_pred - u_bc2))


def initial_condition_loss(model, X_ic, u_ic):
    u_pred = model(X_ic)
    return tf.reduce_mean(tf.square(u_pred - u_ic))




## Loss mediated by Physics information

def pinn_loss(model, X_data, y_data, X_pde):
    # Data loss (e.g., mean squared error)
    y_pred = model(X_data)
    data_loss = tf.reduce_mean(tf.square(y_pred - y_data))

    # Physics loss (PDE residuals)
    pde_residual = compute_pde_residual(model, X_pde)
    physics_loss = tf.reduce_mean(tf.square(pde_residual))

    # Total loss
    total_loss = data_loss + physics_loss
    return total_loss





## Train the model

# Convert to tensors
X_data = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_data = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_pde = tf.convert_to_tensor(x_train, dtype=tf.float32)


#X_data = tf.convert_to_tensor(X_u_train, dtype=tf.float32)
#y_data = tf.convert_to_tensor(u_train, dtype=tf.float32)
#X_pde = tf.convert_to_tensor(X_u_train, dtype=tf.float32)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Custom training loop
@tf.function
def train_step(model, X_data, y_data, X_pde):
    with tf.GradientTape() as tape:
        loss_pde = pinn_loss(model, X_data, y_data, X_pde)
        loss_bc1 = boundary_condition_loss1(model, X_bc1, u_bc1)
        loss_bc2 = boundary_condition_loss2(model, X_bc2, u_bc2)
        #loss_ic = initial_condition_loss(model, X_ic, u_ic)
        loss = loss_pde + loss_bc1 + loss_bc2 #+ loss_ic
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training the model
epochs = 1500
for epoch in range(epochs):
    loss = train_step(model, X_data, y_data, X_pde)
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
        
        
        
        
## Now predict

Y_pred_final = model.predict(X_star)

u_star1 = u_star.reshape(1000, 100)

Y_pred_final1 = Y_pred_final.reshape(1000, 100)


## Do the contour plot

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

cp1 = axs[0].contourf(X, T, u_star1)
axs[0].clabel(cp1, inline=True, fontsize=10)
axs[0].set_title('Contour Plot of u_star')
axs[0].set_xlabel('X')
axs[0].set_ylabel('T')


cp2 = axs[1].contourf(X, T, Y_pred_final1)
axs[1].clabel(cp2, inline=True, fontsize=10)
axs[1].set_title('Contour Plot of y_pred')
axs[1].set_xlabel('X')
axs[1].set_ylabel('T')

# Adjust layout
plt.tight_layout()

# Display the plots
plt.show()


## Do the line plot

fig, axs = plt.subplots(1, 3, figsize=(16, 6))

axs[0].plot(x, u_star1[0], linestyle = '-', label = 'Exact', linewidth = 6.0 )
axs[0].plot(x, Y_pred_final1[0], linestyle = '--', label = 'Predicted')
axs[0].set_title(r'$t = 0.1$')
axs[0].set_xlabel(r'$x$')
axs[0].set_ylabel(r'$u(x, t)$')

axs[1].plot(x, u_star1[749], linestyle = '-', label = 'Exact', linewidth = 6.0 )
axs[1].plot(x, Y_pred_final1[749], linestyle = '--', label = 'Predicted')
axs[1].set_title(r'$t = 7.5$')
axs[1].set_xlabel(r'$x$')
axs[1].set_ylabel(r'$u(x, t)$')


axs[2].plot(x, u_star1[999], linestyle = '-', label = 'Exact', linewidth = 6.0 )
axs[2].plot(x, Y_pred_final1[999], linestyle = '--', label = 'Predicted')
axs[2].set_title(r'$t = 10$')
axs[2].set_xlabel(r'$x$')
axs[2].set_ylabel(r'$u(x, t)$')


for ax in axs:
    ax.set_ylim(0, 1.0)
    ax.legend()

plt.tight_layout()
plt.show()
        
















