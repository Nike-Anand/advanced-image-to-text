import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from ocr_tamil.strhub.models.parseq.system import PARSeq
from ocr_tamil.strhub.data.utils import Tokenizer, CharsetAdapter
from charset_utils import CHARSET_MULTILANG


# ===============================
# DATASET
# ===============================

class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".jpg"):
                    img_path = os.path.join(root, f)
                    txt_path = img_path.replace(".jpg", ".txt")
                    if os.path.exists(txt_path):
                        self.image_paths.append(img_path)
                        with open(txt_path, "r", encoding="utf-8") as t:
                            self.labels.append(t.read().strip())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ===============================
# LIGHTNING MODULE
# ===============================

class PARSeqLightning(pl.LightningModule):
    def __init__(self, charset, lr=3e-4, batch_size=8):
        super().__init__()
        self.save_hyperparameters()

        self.charset = charset
        self.tokenizer = Tokenizer(charset)

        charset_train = ''.join(charset)
        charset_test = ''.join(charset)  # Using same charset for train and test

        self.model = PARSeq(
            charset_train=charset_train,
            charset_test=charset_test,
            max_label_length=25,
            batch_size=batch_size,
            lr=lr,
            warmup_pct=0.075,
            weight_decay=0.0,
            img_size=[32, 128],
            patch_size=[4, 8],
            embed_dim=384,
            enc_num_heads=6,
            enc_mlp_ratio=4,
            enc_depth=12,
            dec_num_heads=12,
            dec_mlp_ratio=4,
            dec_depth=1,
            perm_num=6,
            perm_forward=True,
            perm_mirrored=True,
            decode_ar=True,
            refine_iters=1,
            dropout=0.1
        )

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.model.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.model.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.model.pad_id) | (tgt_in == self.model.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.model.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.model.generate_attn_masks(perm)
            out = self.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.model.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.model.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.model.eos_id, self.model.pad_id, tgt_out)
                n = (tgt_out != self.model.pad_id).sum().item()
        loss /= loss_numel

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-4)


# ===============================
# MAIN TRAINING SCRIPT
# ===============================

def main():
    print("🚀 Starting Multi-Language PARSeq Training")

    # ---- CONFIG ----
    TRAIN_DIR = os.path.join(os.path.dirname(__file__), "datasets", "multilang_train")
    VAL_DIR = os.path.join(os.path.dirname(__file__), "datasets", "multilang_val")
    BATCH_SIZE = 8
    EPOCHS = 50

    charset = list(CHARSET_MULTILANG)

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_ds = OCRDataset(TRAIN_DIR, transform)
    val_ds = OCRDataset(VAL_DIR, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = PARSeqLightning(charset, batch_size=BATCH_SIZE)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=EPOCHS,
        precision="16-mixed",
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/parseq_multilang_final.pt")
    print("✅ Training complete & model saved!")


if __name__ == "__main__":
    main()
