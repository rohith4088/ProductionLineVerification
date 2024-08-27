# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer



# model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
#     attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
# model = model.eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

# image1 = Image.open('research/few_shots.py').convert('RGB')
# image2 = Image.open('resources/IMG_0812.jpg').convert('RGB')
# question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'

# msgs = [{'role': 'user', 'content': [image1, image2, question]}]

# answer = model.chat(
#     image=None,
#     msgs=msgs,
#     tokenizer=tokenizer
# )
# print(answer)

