import torch, PIL.Image as Image
from transformers import AutoProcessor, AutoModel

mid = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(mid).eval()
proc  = AutoProcessor.from_pretrained(mid)

img = Image.open("cat.jpg").convert("RGB")
texts = ["a photo of a cat", "a photo of a dog"]

batch = proc(text=texts, images=img, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**batch)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
    sims = (img_emb @ txt_emb.T).squeeze()

print(list(zip(texts, sims.tolist())))