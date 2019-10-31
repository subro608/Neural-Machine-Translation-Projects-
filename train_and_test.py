from model import Encoder, Decoder, Seq2Seq
from model2 import Encoder2, Decoder2, Seq2Seq2, Attention
from data import get_data
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, PAD_IDX, clip):
    model.train()
    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, PAD_IDX):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    train_iterator, valid_iterator, test_iterator, INPUT_DIM, OUTPUT_DIM, PAD_IDX = get_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 6
    CLIP = 1
    best_valid_loss = float('inf')
    choice = input('which model you want to use?\n')
    if choice == 'seq2seq':
        print('not attention')

        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        model = Seq2Seq(enc, dec, device).to(device)

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, PAD_IDX, CLIP)
            valid_loss = evaluate(model, valid_iterator, PAD_IDX)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model1.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    elif choice == 'seq2seq_with_attention':
        print('with attention')
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = Encoder2(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder2(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

        model2 = Seq2Seq2(enc, dec, device).to(device)
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model2, train_iterator, PAD_IDX, CLIP)
            valid_loss = evaluate(model2, valid_iterator, PAD_IDX)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model2.state_dict(), 'model2.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    prediction_choice = input('Do you want to test predictions?\n')
    if prediction_choice == 'yes':
        model_name = input('On which model would you like to test the predictions?\n')

        if model_name == 'seq2seq':
            model.load_state_dict(torch.load('model1.pt'))
            test_loss = evaluate(model, test_iterator, PAD_IDX)
            print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        if model_name == 'attention_seq2seq':
            model2.load_state_dict(torch.load('model2.pt'))
            test_loss = evaluate(model2, test_iterator, PAD_IDX)
            print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


if __name__ == '__main__':
    main()
