import sys
import os
import argparse
import torch
from transformers import BertTokenizer
from models import NeurTxt
import time
import json
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, m, yb) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        x = x.to(device)
        m = m.to(device)
        yb = yb.to(device)

        # compute output
        output = model(x, m)
        loss = criterion(output, yb)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # Record loss
        losses.update(loss.item(), x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (x, m, yb) in enumerate(val_loader):

            x = x.to(device)
            m = m.to(device)
            yb = yb.to(device)

            # compute output
            output = model(x, m)
            loss = criterion(output, yb)

            output = output.float()
            loss = loss.float()

            # record loss
            losses.update(loss.item(), x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    print('Validation\t  Loss@1: {loss.avg:.3f}\n'
          .format(loss=losses))

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PART', type=int, help='Specify the section/part of the test')
commandLineParser.add_argument('DATA', type=str, help='Specify input txt file with useful data')
commandLineParser.add_argument('OUT', type=str, help='Specify the output pth to save model parameters')
commandLineParser.add_argument('GRADES', type=str, help='Specify grades file')
commandLineParser.add_argument('SEED', type=int, help='Specify training seed')

args = commandLineParser.parse_args()
part = args.PART
data_file = args.DATA
out_file = args.OUT
grades_file = args.GRADES
seed = args.SEED
torch.manual_seed(seed)

# Get the device
device = get_default_device()

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/train.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# Load the data
with open(data_file, 'r') as f:
    utterances = json.loads(f.read())
print("Loaded Data")

# Convert json output from unicode to string
utterances = [[str(item[0]), str(item[1])] for item in utterances]

# Concatentate utterances of a speaker
spk_to_utt = {}
for item in utterances:
    fileName = item[0]
    speakerid = fileName[:12]
    sentence = item[1]

    if speakerid not in spk_to_utt:
        spk_to_utt[speakerid] =  sentence
    else:
        spk_to_utt[speakerid] =spk_to_utt[speakerid] + ' ' + sentence

# get speakerid to section grade dict
grade_dict = {}

lines = [line.rstrip('\n') for line in open(grades_file)]
for line in lines[1:]:
        speaker_id = line[:12]
        grade_overall = line[-3:]
        grade1 = line[-23:-20]
        grade2 = line[-19:-16]
        grade3 = line[-15:-12]
        grade4 = line[-11:-8]
        grade5 = line[-7:-4]
        grades = [grade1, grade2, grade3, grade4, grade5, grade_overall]

        grade = float(grades[part-1])
        grade_dict[speaker_id] = grade

# Create list of grades and speaker utterance in same speaker order
grades = []
vals = []

for id in spk_to_utt:
    try:
        grades.append(grade_dict[id])
        vals.append(spk_to_utt[id])
    except:
        print("Falied for speaker " + str(id))

# Tokenize text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(vals, padding=True, truncation=True, return_tensors="pt")
X = encoded_inputs['input_ids']
mask = encoded_inputs['attention_mask']

y = torch.FloatTensor(grades)

# Separate into training and validation set
validation_size = 100
X_train = X[validation_size:]
X_val = X[:validation_size]
mask_train = mask[validation_size:]
mask_val = mask[:validation_size]
y_train = y[validation_size:]
y_val = y[:validation_size]

# Store as a dataset
train_ds = TensorDataset(X_train, mask_train, y_train)
val_ds = TensorDataset(X_val, mask_val, y_val)

# Use dataloader to handle batches easily
batch_size = 50
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

# Training algorithm
lr = 8*1e-4
epochs = 20
sch = 0.985

model = NeurTxt()
model.to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

for epoch in range(epochs):
    # train for one epoch
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train(train_dl, model, criterion, optimizer, epoch)
    scheduler.step()

    # Evaluate on validation set
    validate(val_dl, model, criterion)

# Save the model
state = model.state_dict()
torch.save(state, out_file)
