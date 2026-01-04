import torch 
import random
import time 
from torch.utils.data import DataLoader, RandomSampler



def create_length_grouped_video_text_dataloader(dataset, batch_size, num_workers, max_frames, 
   epoch=0):
    
    sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=identity,
        drop_last=True,
    )

    # make it infinite
    dataloader_iterator = TemporalLengthBucketeer(
        dataloader,
        max_frames=max_frames,
        epoch=epoch,
    )

    return dataloader_iterator

class TemporalLengthBucketeer:
    def __init__(
        self, dataloader, max_frames=16, epoch=0,
    ):
        self.batch_size = dataloader.batch_size
        self._dataloader = dataloader
        self.iterator = iter(dataloader)
        self.buckets = {temp: [] for temp in range(1, max_frames + 1)}
        self._epoch = epoch

    def get_available_batch(self):
        available_size = []
        for b in self.buckets:
            if len(self.buckets[b]) >= self.batch_size:
                available_size.append(b)

        if len(available_size) == 0:
            return None
        else:
            b = random.choice(available_size)
            batch = self.buckets[b][:self.batch_size]
            self.buckets[b] = self.buckets[b][self.batch_size:]
            return batch

    def __next__(self):
        batch = self.get_available_batch()
        while batch is None:
            try:
                elements = next(self.iterator)
            except StopIteration:
                # To make it infinity
                self._epoch += 1
                if hasattr(self._dataloader.sampler, "set_epoch"):
                    self._dataloader.sampler.set_epoch(self._epoch)
                time.sleep(2) # Prevent possible deadlock during epoch transition
                self.iterator = iter(self._dataloader)
                elements = next(self.iterator)

            for dct in elements:
                try:
                    video_latent = dct['video']
                    temp = video_latent.shape[2]
                    self.buckets[temp].append({**{'video': video_latent}, **{k:dct[k] for k in dct if k != 'video'}})
                except Exception as e:
                    continue

            batch = self.get_available_batch()

        out = {k:[batch[i][k] for i in range(len(batch))] for k in batch[0]}
        out = {k: torch.cat(o, dim=0) if isinstance(o[0], torch.Tensor) else o for k, o in out.items()}

        if 'prompt_embed' in out:
            # Loading the pre-extrcted textual features
            prompt_embeds = out['prompt_embed'].clone()
            del out['prompt_embed']
            prompt_attention_mask = out['prompt_attention_mask'].clone()
            del out['prompt_attention_mask']
            pooled_prompt_embeds = out['pooled_prompt_embed'].clone()
            del out['pooled_prompt_embed']

            out['text'] = {
                'prompt_embeds' : prompt_embeds,
                'prompt_attention_mask': prompt_attention_mask,
                'pooled_prompt_embeds': pooled_prompt_embeds,
            }

        return out

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterator)
    

def identity(x):
    return x
