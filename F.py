import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
from Classes import *


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def train_neural_network(inputs, targets):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


def predict(input_data):
    model.eval()
    with torch.no_grad():
        return model(input_data)


input_size = 2  # Adjust input size based on your data
output_size = 1  # Adjust output size based on your data
model = SimpleNN(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)



root = tk.Tk()
root.title("Space Invaders")
root.geometry("500x500")
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)


while run:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player.move(-speed, 0)
    if keys[pygame.K_RIGHT]:
        player.move(speed, 0)
    if keys[pygame.K_SPACE] and shoot_delay == 0:
        bullets.append(Bullet(player.x + player.width // 2, player.y, 5, 5, 1, (255, 255, 255), 5))
        shoot_delay = 1

        # Example of using the neural network
        input_data = torch.tensor([player.x, player.y], dtype=torch.float32)
        prediction = predict(input_data)
        print("Neural Network Prediction:", prediction.item())

    shoot_delay = (shoot_delay + (shoot_delay != 0)) % 7

    canvas.delete("all")  # Clear canvas for GUI

    draw_army(army)
    draw_bullets(bullets)

    army.move(speed, 20, 500 - 20, player.y)
    move_bullets(bullets)

    player.draw(canvas)

    root.update_idletasks()
    root.update()

pygame.quit()
