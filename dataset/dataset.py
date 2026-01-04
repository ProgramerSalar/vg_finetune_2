import torch
import jsonlines
from tqdm import tqdm
from torch.utils.data import Dataset
import random



class LengthGroupedVideoTextDataset(Dataset):
    """
        Usage:
            The dataset class for video-text pairs, used for video generation training
            It groups the video with the same frames together
            Now only supporting fixed resolution during training
        params:
            anno_file: The annotation file list
            max_frames: The maximum temporal lengths (This is the vae latent temporal length) 16 => (16 - 1) * 8 + 1 = 121 frames
            load_vae_latent: Loading the pre-extracted vae latents during training, we recommend to extract the latents in advance
                to reduce the time cost per batch
            load_text_fea: Loading the pre-extracted text features during training, we recommend to extract the prompt textual features
                in advance, since the T5 encoder will cost many GPU memories
    """
    
    def __init__(self, anno_file, max_frames=16, resolution='384p', load_vae_latent=True, load_text_fea=True):
        super().__init__()

        self.video_annos = []
        self.max_frames = max_frames
        self.load_vae_latent = load_vae_latent
        self.load_text_fea = load_text_fea
        self.resolution = resolution

        assert load_vae_latent, "Now only support loading vae latents, we will support to directly load video frames in the future"

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        for anno_file_ in anno_file:
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)
        
        print(f"Totally Remained {len(self.video_annos)} videos") 

    def __len__(self):
        return len(self.video_annos)

    def __getitem__(self, index):
        try:
            video_anno = self.video_annos[index]
            text = video_anno['text']
            latent_path = video_anno['video_latent']
            latent = torch.load(latent_path, map_location='cpu')  # loading the pre-extracted video latents

            # TODO: remove the hard code latent shape checking
            if self.resolution == '384p':
                assert latent.shape[-1] == 640 // 8
                assert latent.shape[-2] == 384 // 8
            else:
                assert self.resolution == '768p'
                assert latent.shape[-1] == 1280 // 8
                assert latent.shape[-2] == 768 // 8

            cur_temp = latent.shape[2]
            cur_temp = min(cur_temp, self.max_frames)

            video_latent = latent[:,:,:cur_temp].float()
            assert video_latent.shape[1] == 16

            if self.load_text_fea:
                text_fea_path = video_anno['text_fea']
                text_fea = torch.load(text_fea_path, map_location='cpu')
                return {
                    'video': video_latent,
                    'prompt_embed': text_fea['prompt_embed'],
                    'prompt_attention_mask': text_fea['prompt_attention_mask'],
                    'pooled_prompt_embed': text_fea['pooled_prompt_embed'],
                    "identifier": 'video',
                }

            else:
                return {
                    'video': video_latent,
                    'text': text,
                    "identifier": 'video',
                }

        except Exception as e:
            print(f'Load Video Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))