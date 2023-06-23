import torch
import torch.nn as nn
import string
import glob
import unicodedata
import os
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Define some helper functions
def findFiles(path): return glob.glob(path)
def unicodeToAscii(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)
def letterToIndex(letter): return all_letters.find(letter)
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
def randomChoice(l): return l[random.randint(0, len(l) - 1)]
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# Load data
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)

# Create the RNN
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# Training
n_iters = 100000
print_every = 5000
plot_every = 1000
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.005)

# Keep track of losses for plotting
current_loss = 0
all_losses = []

# Keep track of accuracy for plotting
correct_predictions = 0
all_accuracy = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    input = lineToTensor(line)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(input.size()[0]):
        output, hidden = rnn(input[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    current_loss += loss.item()
    if iter % print_every == 0:
        print('%d %d%% %.4f' % (iter, iter / n_iters * 100, loss))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

    # Compute accuracy
    guess, _ = categoryFromOutput(output)
    correct_predictions += (guess == category)
    if iter % plot_every == 0:
        accuracy = correct_predictions / plot_every
        all_accuracy.append(accuracy)
        correct_predictions = 0

# Plotting the results
plt.figure()
plt.plot(all_losses)
plt.title('Loss over time')
plt.savefig('loss.png')

plt.figure()
plt.plot(all_accuracy)
plt.title('Accuracy over time')
plt.savefig('accuracy.png')

# Confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('confusion.png')
plt.show()
