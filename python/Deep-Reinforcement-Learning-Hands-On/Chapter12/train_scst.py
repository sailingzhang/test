#!/usr/bin/env python3

import sys
sys.path.append("../../")
sys.path.append("../../ptan-master")
import logging as log
from log_init import log_init
import os
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter

from libbots import data, model, utils

import torch
import torch.optim as optim
import torch.nn.functional as F

import ptan

SAVES_DIR = "saves"

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MAX_EPOCHES = 10000




# log = logging.getLogger("train")


def run_test(test_data, net, end_token, device="cpu"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, net.emb, device)
        enc = net.encode(input_seq)
        _, tokens = net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                            stop_at_token=end_token)
        ref_indices = [
            indices[1:]
            for indices in p2
        ]
        bleu_sum += utils.calc_bleu_many(tokens, ref_indices)
        bleu_count += 1
    return bleu_sum / bleu_count


if __name__ == "__main__":
    log_init("../../12_train_scst.log")
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Category to use for training. Empty string to train on full dataset")
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", required=True, help="Load model and continue in RL mode")
    parser.add_argument("--samples", type=int, default=4, help="Count of samples in prob mode")
    parser.add_argument("--disable-skip", default=False, action='store_true', help="Disable skipping of samples with high argmax BLEU")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    phrase_pairs, emb_dict = data.load_data(genre_filter=args.data)
    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data)
    log.info("Training data converted, got %d samples", len(train_data))
    train_data = data.group_train_data(train_data)
    test_data = data.group_train_data(test_data)
    # logging.debug("train_data={}".format(train_data))#这是一对多。
    #train_data= [((1, 162, 13, 134, 578, 31, 2), [[1, 44, 14, 36, 14, 67, 43, 1228, 27, 9, 3415, 31, 2]]), ((1, 252, 46, 172, 4, 876, 299, 173, 2), [[1, 62, 31, 275, 13, 1931, 96, 25, 162, 43, 31, 2]])]
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    log.info("Model: %s", net)

    writer = SummaryWriter(comment="-" + args.name)
    net.load_state_dict(torch.load(args.load))
    log.info("Model loaded from %s, continue training in RL mode...", args.load)

    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)

    with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
        optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
        batch_idx = 0
        best_bleu = None
        for epoch in range(MAX_EPOCHES):
            random.shuffle(train_data)
            dial_shown = False

            total_samples = 0
            skipped_samples = 0
            bleus_argmax = []
            bleus_sample = []

            for batch in data.iterate_batches(train_data, BATCH_SIZE):
                batch_idx += 1
                optimiser.zero_grad()
                input_seq, input_batch, output_batch = model.pack_batch_no_out(batch, net.emb, device)
                # logging.debug("input_batch={},output_batch={}".format(input_batch,output_batch))
                # DEBUG input_patch=((1, 88, 555, 93, 16, 3072, 123, 371, 2836, 13, 60, 797, 3685, 3416, 13, 551, 197, 14, 2), (1, 5, 1588, 46, 176, 14, 5, 6, 13, 83, 241, 93, 488, 51, 94, 155, 43, 14, 2), (1, 195, 918, 35, 1372, 173, 2), (1, 1512, 4, 2202, 14, 2))
                # output_batch=([[1, 21, 22, 366, 13, 230, 16, 493, 31, 2]], [[1, 408, 745, 93, 14, 2]], [[1, 2285, 14, 114, 51, 77, 3820, 69, 276, 177, 1219, 46, 620, 390, 14, 2]], [[1, 252, 20, 2]], [[1, 91, 417, 4267, 353, 14, 2]],  [[1, 25, 309, 14, 2]], [[1, 275, 13, 1252, 31, 31, 2]], [[1, 387, 31, 2]], [[1, 658, 14, 2]], [[1, 105, 43, 67, 14, 2]],[[1, 387, 14, 2]], [[1, 1736, 4, 626, 173, 2]], [[1, 45, 663, 4, 2178, 14, 2]])
                enc = net.encode(input_seq)

                net_policies = []
                net_actions = []
                net_advantages = []
                beg_embedding = net.emb(beg_token)

                for idx, inp_idx in enumerate(input_batch):
                    total_samples += 1
                    ref_indices = [
                        indices[1:]
                        for indices in output_batch[idx]
                    ]
                    item_enc = net.get_encoded_item(enc, idx)
                    r_argmax, actions = net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                stop_at_token=end_token)
                    # logging.debug("r_argmax={},actions={}".format(r_argmax,actions))
    #      r_argmax=tensor([[-13.8743, -13.3444,  -0.6508,  ...,  -9.0594,  -8.2211, -13.8162],
    #     [-12.8510, -12.9394,   0.0347,  ..., -11.4010,  -5.9313, -12.9519],
    #     [-10.5749, -10.3025,   3.0768,  ...,  -7.3702,  -6.2836, -10.4178],
    #     [-15.4379, -15.1566,   4.2264,  ..., -12.3431,  -5.4633, -15.4407],
    #     [-13.7221, -13.5014,   5.6320,  ..., -11.3432,  -5.0856, -13.8136]],
    #    grad_fn=<CatBackward>),actions=[5, 207, 146, 14, 2]
                    argmax_bleu = utils.calc_bleu_many(actions, ref_indices)
                    bleus_argmax.append(argmax_bleu)

                    if not args.disable_skip and argmax_bleu > 0.99:
                        skipped_samples += 1
                        continue

                    if not dial_shown:
                        log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, rev_emb_dict)))
                        ref_words = [utils.untokenize(data.decode_words(ref, rev_emb_dict)) for ref in ref_indices]
                        log.info("Refer: %s", " ~~|~~ ".join(ref_words))
                        log.info("Argmax: %s, bleu=%.4f", utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                                 argmax_bleu)

                    for _ in range(args.samples):
                        r_sample, actions = net.decode_chain_sampling(item_enc, beg_embedding,
                                                                      data.MAX_TOKENS, stop_at_token=end_token)
                        sample_bleu = utils.calc_bleu_many(actions, ref_indices)

                        if not dial_shown:
                            log.info("Sample: %s, bleu=%.4f", utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                                     sample_bleu)

                        net_policies.append(r_sample)
                        net_actions.extend(actions)
                        net_advantages.extend([sample_bleu - argmax_bleu] * len(actions))
                        bleus_sample.append(sample_bleu)
                        # logging.debug("r_sample={},actions={},sample_bleu - argmax_bleu={},sample_bleu={}".format(r_sample,actions,sample_bleu - argmax_bleu,sample_bleu))
                        # r_sample=tensor([[-11.8627, -11.6143,   0.1806,  ...,  -7.5314,  -7.5344, -12.2520],
                        #                 [-13.6776, -13.7746,   1.1900,  ..., -12.3752,  -9.5009, -14.1215],
                        #                 [-18.6475, -18.5967,   2.3067,  ..., -11.8892, -14.9023, -18.9035],
                        #                 ...,
                        #                 [-13.8555, -13.4120,   2.7234,  ...,  -8.7176,  -7.8477, -13.8406],
                        #                 [-16.6379, -15.8786,   3.4330,  ..., -14.6808,  -9.7412, -16.6886],
                        #                 [-12.5753, -12.2687,   9.0037,  ...,  -9.5653,  -6.3280, -12.6913]],
                        #             grad_fn=<CatBackward>),actions=[44, 171, 13, 14, 292, 64, 13, 241, 93, 108, 31, 2],sample_bleu - argmax_bleu=0.030587696549717394,sample_bleu=0.12309149097933275

                    dial_shown = True

                if not net_policies:
                    continue

                policies_v = torch.cat(net_policies)
                actions_t = torch.LongTensor(net_actions).to(device)
                adv_v = torch.FloatTensor(net_advantages).to(device)
                log_prob_v = F.log_softmax(policies_v, dim=1)
                log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                loss_v = loss_policy_v
                loss_v.backward()
                optimiser.step()

                logging.debug("policies_v={},log_prob_v={}".format(policies_v,log_prob_v))
# DEBUG policies_v=tensor([[-15.4101, -14.7879,  -1.3527,  ...,  -9.7199,  -9.6832, -15.6094],
#         [-11.4530, -11.6235,  -0.2077,  ..., -10.1832,  -6.4477, -11.5834],
#         [-11.2781, -11.0824,   2.4559,  ...,  -7.4973,  -9.5905, -11.4410],
#         ...,
#         [ -8.1145,  -7.6796,  -0.3306,  ...,  -8.2129,  -8.0966,  -8.2438],
#         [-11.2506, -10.6323,   1.8057,  ...,  -9.3147,  -9.4105, -11.4355],
#         [-15.9063, -15.4262,   9.4001,  ...,  -9.7805,  -9.1733, -16.2098]],
#        grad_fn=<CatBackward>),log_prob_v=tensor([[-23.4501, -22.8279,  -9.3927,  ..., -17.7598, -17.7231, -23.6493],
#         [-18.7960, -18.9666,  -7.5507,  ..., -17.5262, -13.7907, -18.9264],
#         [-17.9625, -17.7668,  -4.2284,  ..., -14.1816, -16.2748, -18.1253],
#         ...,
#         [-14.6695, -14.2346,  -6.8856,  ..., -14.7679, -14.6516, -14.7988],
#         [-18.2389, -17.6206,  -5.1826,  ..., -16.3030, -16.3988, -18.4238],
#         [-25.3614, -24.8813,  -0.0550,  ..., -19.2356, -18.6284, -25.6649]],
#        grad_fn=<LogSoftmaxBackward>)


                tb_tracker.track("advantage", adv_v, batch_idx)
                tb_tracker.track("loss_policy", loss_policy_v, batch_idx)
                tb_tracker.track("loss_total", loss_v, batch_idx)

            bleu_test = run_test(test_data, net, end_token, device)
            bleu = np.mean(bleus_argmax)
            writer.add_scalar("bleu_test", bleu_test, batch_idx)
            writer.add_scalar("bleu_argmax", bleu, batch_idx)
            writer.add_scalar("bleu_sample", np.mean(bleus_sample), batch_idx)
            writer.add_scalar("skipped_samples", skipped_samples / total_samples, batch_idx)
            writer.add_scalar("epoch", batch_idx, epoch)
            log.info("Epoch %d, test BLEU: %.3f", epoch, bleu_test)
            if best_bleu is None or best_bleu < bleu_test:
                best_bleu = bleu_test
                log.info("Best bleu updated: %.4f", bleu_test)
                torch.save(net.state_dict(), os.path.join(saves_path, "bleu_%.3f_%02d.dat" % (bleu_test, epoch)))
            if epoch % 10 == 0:
                torch.save(net.state_dict(), os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu, bleu_test)))

    writer.close()
