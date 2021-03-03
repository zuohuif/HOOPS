import time
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def load_dialog_utter_resp(filename):
    data = np.load(filename)
    utter = data["utterance"].astype(np.int64)
    response = data["response"].astype(np.int64)
    label = data["label"].astype(np.float32)
    return utter, response, label


def load_dialog_word_embed(filename):
    obj = None
    if filename.endswith('.pkl'):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        obj = np.array(obj)
    elif filename.endswith('.npz'):
        data = np.load(filename)
        obj = data['word_emb'].astype(np.float32)
    return obj


def load_dialog_gt_context(filename):
    data = np.load(filename)
    utter_gt = data["utter_gt"].astype(np.int64).squeeze(axis=-1)
    utter_context = data["utter_context"].astype(np.int64)
    resp_gt = data["resp_gt"].astype(np.int64).squeeze(axis=-1)
    return utter_gt, utter_context, resp_gt


def load_rec_train(filename):
    data = np.loadtxt(filename, delimiter="\t", dtype=np.int).astype(np.int64)
    return data


def load_rec_candidates(filename):
    data = np.load(filename)
    return data["candidates"]


def load_kg_entities(filename):
    eid2ent = {}
    with open(filename, "r") as f:
        for line in f:
            tmp = line.split("\t")
            eid2ent[int(tmp[1])] = tmp[0]
    return eid2ent


def load_kg_embeddings(filename):
    ckpt = torch.load(filename)
    embeds = ckpt["ent_embeddings.weight"].cpu()
    rels = ckpt["rel_embeddings.weight"].cpu()
    return embeds, rels


class ConvRecKGDataset(Dataset):
    """Conversational recommendation dataset."""

    def __init__(self, args, num_neg_items=1, is_train=True):
        super(ConvRecKGDataset, self).__init__()
        self.args = args
        self.num_neg_items = num_neg_items
        self.is_train = is_train
        self._load_dialog_data()
        self._load_recommend_data()
        self._load_kg_data()

    def _load_dialog_data(self):
        print("Loading dialog data...")
        t1 = time.time()

        # In training, #utter:#resp=1:2 (positive/negative).
        # In test, #utter:#resp=1:10 (pick 1 out of 10 candidates).
        if self.is_train:
            self.x_utter, self.x_response, self.y_label = load_dialog_utter_resp(self.args.dial_utter_resp_train_file)
            self.utter_gt, self.utter_context, self.resp_gt = load_dialog_gt_context(self.args.dial_gt_context_train_file)
        else:
            self.x_utter, self.x_response, self.y_label = load_dialog_utter_resp(self.args.dial_utter_resp_test_file)
            self.utter_gt, self.utter_context, self.resp_gt = load_dialog_gt_context(self.args.dial_gt_context_test_file)
        bs = self.x_utter.shape[0]
        ratio = len(self.y_label) // bs
        self.x_response = self.x_response.reshape(bs, ratio, -1)
        self.y_label = self.y_label.reshape(bs, ratio)
        self.resp_gt = self.resp_gt.reshape(bs, ratio)
        print("  Utterrances:", self.x_utter.shape)
        print("  Responses:", self.x_response.shape)
        print("  Labels:", self.y_label.shape)
        print("  Utter-Gt:", self.utter_gt.shape)
        print("  Utter-Context:", self.utter_context.shape)
        print("  Response-Gt:", self.resp_gt.shape)

        # Load vocabulary and word embeddings
        self.word_embed = load_dialog_word_embed(self.args.dial_word_embed_file)
        print("  Word embeddings:", self.word_embed.shape)

        t2 = time.time()
        print("Takes {:.4f}s!".format(t2 - t1))

    def _load_recommend_data(self):
        print("Loading recommendation data...")
        t1 = time.time()

        if self.is_train:
            # np.array [#user-item-pairs, 2], where column 0 is user, 1 is positive item.
            self.rec_data = load_rec_train(self.args.rec_train_file)

            self.purchase_hist = {}
            for row in self.rec_data:
                uid, iid = row[0], row[1]
                if uid not in self.purchase_hist:
                    self.purchase_hist[uid] = set()
                self.purchase_hist[uid].add(iid)
            # print("  #users:", len(self.purchase_hist))
        else:
            # np.array [#user-item-pairs, 102], where column 0 is user, 1 is positive item, 2-101 are negative items.
            self.rec_data = load_rec_candidates(self.args.rec_test_candidate_file)
        print("  User-item data:", self.rec_data.shape)


        t2 = time.time()
        print("Takes {:.4f}s!".format(t2 - t1))

    def _load_kg_data(self):
        print("Loading KG data...")
        t1 = time.time()

        self.users = load_kg_entities(self.args.kg_user_entities_file)  # dict(key=id, value=name)
        self.num_users = len(self.users)
        print(
            "  #users: {} (min={}, max={})".format(
                self.num_users, min(list(self.users.keys())), max(list(self.users.keys()))
            )
        )
        self.items = load_kg_entities(self.args.kg_item_entities_file)
        self.num_items = len(self.items)
        print(
            "  #items: {} (min={}, max={})".format(
                self.num_items, min(list(self.items.keys())), max(list(self.items.keys()))
            )
        )

        # Note: Here we assume item IDs are right after user IDs.
        assert min(list(self.items.keys())) == len(self.users)
        assert max(list(self.items.keys())) == len(self.users) + len(self.items) - 1

        self.entity_embed, self.rel_embed = load_kg_embeddings(self.args.kg_transe_embed_file)
        print("  entity embed: {}".format(self.entity_embed.size()))
        print("  relation embed: {}".format(self.rel_embed.size()))

        t2 = time.time()
        print("Takes {:.4f}s!".format(t2 - t1))

    def neg_sample(self, user_id, item_id):
        # Note: Here we randomly sample an item as negative one, since the
        # probability of sampling a positive item is very low.
        # TODO: A better sampling method can be adopted here.
        #neg_items = np.random.randint(
        #    len(self.users), len(self.users) + len(self.items), size=self.num_neg_items
        #).astype(np.int64)

        assert self.is_train
        pos_iid = self.purchase_hist[user_id]  # set
        start_idx = self.num_users
        end_idx = self.num_users + self.num_items
        neg_items = np.random.randint(start_idx, end_idx, size=self.num_neg_items + len(pos_iid))
        neg_items = [i for i in neg_items if i not in pos_iid]
        neg_items = np.array(neg_items[:self.num_neg_items]).astype(np.int64)
        #neg_items = set(neg_items.tolist()) - pos_iid
        #neg_items = np.array(list(neg_items)[:self.num_neg_items]).astype(np.int64)
        #print(pos_iid, neg_items)
        return neg_items

    def __getitem__(self, idx):
        utter = self.x_utter[idx]
        resp = self.x_response[idx]
        label = self.y_label[idx]
        utter_gt = self.utter_gt[idx]
        utter_context = self.utter_context[idx]
        resp_gt = self.resp_gt[idx]
        ui_pair = self.rec_data[idx]
        user = self.rec_data[idx][0]
        items = self.rec_data[idx][1:]
        if self.is_train:
            neg_items = self.neg_sample(user, items)
            #print(items.shape, items)
            items = np.array([items[0]] +  neg_items.tolist())
            #print(items.shape, items)
        # print(type(utter), type(resp), type(label), type(user), type(pos_item),
        #       type(neg_item))
        return utter, resp, label, utter_gt, utter_context, resp_gt, user, items

    def __len__(self):
        return len(self.y_label)
