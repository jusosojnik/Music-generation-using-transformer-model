import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import pypianoroll
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = []
path = '/Users/jusosojnik/PycharmProjects/transformerMusic/Test'
pieces = {}

c = 0
multitrack = None
for i in os.listdir(path):
    print(i)
    multitrack = pypianoroll.read("Test/" + i)
    mat = np.array(multitrack[0].pianoroll)
    pieces[i] = mat
    l = mat.tolist()
    for j in l:
        j = np.array(j)
        j = j[0:128]
        j = list(j)
        vocab.append(str(j))
    c += 1
    if c >= 2: break

vocab = list(set(vocab))
print('number of unique combination of notes ("words"): ' + str(len(vocab)))

w2i = {}
i2w = {}
for i, word in enumerate(vocab):
    w2i[word] = i
    i2w[i] = word


class Dataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, n_pieces=250):
        self.seq_len = seq_len
        self.lengths = {}
        self.examlpe_num = 0

        for piece_name in os.listdir(path)[:n_pieces]:
            piece = pieces[piece_name]
            self.lengths[piece_name] = piece.shape[0] - (self.seq_len + 1)
            self.examlpe_num += piece.shape[0] - (self.seq_len + 1)

    def __len__(self):
        return self.examlpe_num

    def __getitem__(self, example_idx):
        s = 0
        for piece_check in self.lengths:
            if s + self.lengths[piece_check] > example_idx:
                piece = piece_check
                example_idx -= s
                break
            s += self.lengths[piece_check]

        piece = pieces[piece]
        x = piece[example_idx:example_idx + self.seq_len, 0:128]
        y = piece[example_idx + 1:example_idx + self.seq_len + 1, 0:128]

        x = [str(j) for j in x.tolist()]
        y = [str(j) for j in y.tolist()]
        x = [w2i[j] for j in x]
        y = [w2i[j] for j in y]

        x = torch.tensor(x).long()
        y = torch.tensor(y).long()

        y = nn.functional.one_hot(y, num_classes=len(vocab))

        return x, y


class AttentionError(nn.Module, Exception):
    pass


class MultiheadedAttention(nn.Module):
    def __init__(self, d_model, sequence_len, heads=8, dropout=0.1):
        super().__init__()

        if d_model % heads != 0:
            raise AttentionError()

        self.d_model = d_model
        self.heads = heads
        s = d_model // heads

        self.linears = []
        for head in range(heads):
            self.linears.append(nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)]))
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.Er = torch.randn([s, sequence_len], device=device, requires_grad=True)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        b, t, e = x.size()
        h = self.heads
        s = e // h
        x = x.view(b, t, h, s)
        queries, keys, values = [], [], []
        for head in range(self.heads):
            x_head = x[:, :, head, :]
            q, k, v = [w(x) for w, x in zip(self.linears[head].to(device), (x_head, x_head, x_head))]
            queries.append(q)
            keys.append(k)
            values.append(v)

        SRel = []
        for head in range(self.heads):
            QEr = torch.matmul(queries[head], self.Er)
            SRel.append(QEr.contiguous().view(b, t, t))

        head_representations = []
        for head in range(self.heads):
            queries[head] = queries[head] / (e ** (1 / 4))
            keys[head] = keys[head] / (e ** (1 / 4))
            scores_head = torch.bmm(queries[head], keys[head].transpose(1, 2))
            scores = scores_head + SRel[head]
            subsequent_mask = torch.triu(torch.ones(1, t, t, device=device), 1)
            scores = scores.masked_fill(subsequent_mask == 1, -1e9)
            attn_probs = F.softmax(scores, dim=2)
            attn_probs = self.dropout(attn_probs)
            head_representations.append(torch.bmm(attn_probs, values[head]).view(b, t, s))

        out = torch.cat(head_representations, dim=2)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.recombine_heads(out)


class Transformer(nn.Module):
    # m = Transformer(seq_len, len(w2i), emb_head * n_heads, n_heads, n_layers, dropout=0.0).to(device)
    def __init__(self, seq_len, num_token, num_inputs, num_heads, num_layers, dropout=0.3):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.enc = nn.Embedding(num_token, num_inputs - 3)
        self.bns = nn.ModuleList([nn.BatchNorm1d(seq_len) for i in range(num_layers)])
        self.enc_transformer = nn.ModuleList(
            [MultiheadedAttention(num_inputs, seq_len, heads=num_heads, dropout=dropout) for i in range(num_layers)])
        self.num_inputs = num_inputs
        self.dec = nn.Linear(num_inputs, num_token)
        pos_embeds = torch.randn((seq_len, 1, 3), device=device, requires_grad=True)
        self.pos_embeds = pos_embeds.repeat(1, batch_size, 1)

    def forward(self, source):
        source = self.enc(source) * math.sqrt(self.num_inputs)
        source = torch.cat([source, self.pos_embeds], axis=2)

        for layer in range(self.num_layers):
            source = source.swapaxes(0, 1)
            source = self.bns[layer](source)
            source = source.swapaxes(0, 1)
            source = self.enc_transformer[layer](source)
            source = source.swapaxes(0, 1)

        op = self.dec(source)

        return op



if __name__ == '__main__':
    seq_len = 50
    batch_size = 32
    n_heads = 8
    emb_head = 30
    n_layers = 3

    m = Transformer(seq_len, len(w2i), emb_head * n_heads, n_heads, n_layers, dropout=0.0).to(device)
    loss_func = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)

    all_losses = []
    all_accs = []
    # torch.save(m.state_dict(), "Models/model.pt")
    m.train()
    # m = torch.load("Models/model.pt")
    # m.eval()
    # torch.save(m.state_dict(), "Models/model.pt")
    # m.load_state_dict(torch.load("Models/model.pt"))
    for epoch in range(1):
        print('epoch: ' + str(epoch))
        losses = []
        accs = []
        dataset = Dataset(seq_len, 2900)
        generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        counter = 0
        for x, y in generator:
            counter += 1
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            x = x.swapaxes(1, 0)
            y = y.swapaxes(1, 0)
            output = m(x)
            if counter % 5 == 0:
                accs.append(torch.sum(torch.argmax(y, axis=2) == torch.argmax(output, axis=2)) / (
                            output.shape[0] * output.shape[1]))
            loss = loss_func(output, y)
            losses.append(loss.item())
            loss.backward()
            opt.step()
            print(counter)
            if counter % 10 == 0:
                torch.save(m.state_dict(), "Models/model.pt")
                print(f'[{counter}/{len(generator)}]')
                print(round(np.mean(losses) * 10000, 2), round(np.mean([x.cpu().detach().numpy() for x in accs]), 2))
                all_accs.append(round(np.mean([x.cpu().detach().numpy() for x in accs]), 2))
                all_losses.append(round(np.mean(losses) * 10000, 2))
                losses = []
                accs = []
                print()
            if counter == 100: break

    # torch.save(m.state_dict(), "Models/model.pt")
    f, a = plt.subplots(2, 1, figsize=(18, 10))
    print(all_losses)
    print(all_accs)
    a[0].plot(all_losses)
    a[1].plot(all_accs)
    plt.show()

    # m.load_state_dict(torch.load("Models/model.pt"))
    m.eval()
    l = list(range(len(w2i)))
    ln = 100
    temp = 1.3

    mat = pieces[list(pieces.keys())[0]]
    # mat = pieces['MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi']
    gen = mat[:seq_len, :]
    gen = gen[:, :128]
    with torch.no_grad():
        for i in range(ln):
            print(i)
            start = gen[-seq_len:, :]
            start = [str(j) for j in start.tolist()]
            start = [w2i[j] for j in start]
            start = torch.tensor(start).long().to(device).unsqueeze(0)
            start = start.repeat(32, 1)
            start = start.swapaxes(0, 1)

            op = m(start)

            pred = op[-1, 0, :]
            pred = pred.cpu().detach().numpy()
            pred = pred * temp
            pred = softmax(pred)
            pred = np.random.choice(l, p=pred)
            pred = i2w[pred].replace('[', '').replace('[', '')
            pred = np.fromstring(pred, sep=', ').astype(int)
            pred = np.expand_dims(pred, 0)
            gen = np.concatenate([gen, pred])

f, a = plt.subplots(2,1,figsize=(18,10))
mat[mat > 1] = 1
gen_b = gen.copy()
gen_b[gen_b > 0] = 1
gen_b[seq_len,:] = 1
mat[seq_len,:] = 1
print(gen.shape)
a[0].imshow(gen_b.T)
a[1].imshow(mat.T)
# plt.show()

newMultitrack = pypianoroll.Multitrack()
newTrack = pypianoroll.StandardTrack()
newTrack.pianoroll = gen
newMultitrack.tracks.append(newTrack)
pypianoroll.write(path="GeneratedMusic/piece1.midi", multitrack=newMultitrack)