import torch
from tqdm import tqdm
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load mnist dataset
from torchvision import datasets, transforms

mnist_train = datasets.MNIST('data',
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])
                            )

mnist_dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=False)

# load the model
model = torch.load("sempre1cnn4_3999.pt", map_location=device) # it is a sequential
print(model)
model.to(device)

# now finetune model for 1 epoch on mnist
model.train()
# require grads
for param in model.parameters():
    param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1
for epoch in range(num_epochs):
    for data, target in tqdm(mnist_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs} loss: {loss.item()}")

# now test the model on mnist
mnist_test = datasets.MNIST('data',
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])
                            )

mnist_test_dataloader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

model.eval()
correct = 0
with torch.no_grad():
    for data, target in mnist_test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        correct += output.argmax(dim=1).eq(target).sum().item()

print(f"Accuracy on mnist test set: {correct/len(mnist_test)}")