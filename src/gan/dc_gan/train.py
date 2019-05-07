

def train_epoch(model, criterion, data_iterator, device ):
    model.train()
    epoch_loss = 0.0

    for batch_count, batch in enumerate(tqdm(data_iterator)):
        seq, targets, src_lens, tgt_lens = batch
        seq, targets, src_lens, tgt_lens = seq.to(device), targets.to(device), src_lens.to(device), tgt_lens

        optimizer.zero_grad()

        logits = model(seq, targets[:,:-1], src_lens, tgt_lens, teacher_force_ratio)
        loss = criterion(logits.permute(0,2,1), targets[:,1:].long())
        loss = loss.sum().float()/(tgt_lens.sum().float() - len(tgt_lens))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)

        if (batch_count+1) % len(train_loader) == 0:
            print("\n| Epoch: {:02} | Loss: {:0.5f}".format(epoch, loss.item()))
            plot_grad_flow(model.named_parameters())

        optimizer.step()

        levenshtein_dist += calculate_levenshtein(logits, targets[:,1:], tgt_lens)
        epoch_loss += loss.item()

        del  seq, targets
        torch.cuda.empty_cache()

    return epoch_loss / len(data_iterator), levenshtein_dist/ len(data_iterator)
