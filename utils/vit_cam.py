import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import regex as re

from utils.image_utils import show_cam_on_image, show_overlapped_cam


def vit_block_vis(image, target_features, img_encoder, block, device, grad=False, neg_saliency=False):
  
    img_encoder.eval()
    image_features = img_encoder(image)
  

    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features_new = image_features / image_features_norm
    target_features_norm = target_features.norm(dim=-1, keepdim=True)
    target_features_new = target_features / target_features_norm
    
    similarity = image_features_new[0].dot(target_features_new[0])
    image = (image - image.min()) / (image.max() - image.min())

    img_encoder.zero_grad()
    similarity.backward(retain_graph=True)
    
    image_attn_blocks = list(dict(img_encoder.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    
    
    if grad:
        cam = image_attn_blocks[block].attn_grad.detach()
    else:
        cam = image_attn_blocks[block].attn_probs.detach()
        
    cam = cam.mean(dim=0) 
    image_relevance = cam[0, 1:]



    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    
    
    cam = image_relevance * image
    cam = cam / torch.max(cam)
    
    # TODO: maybe we can ignore this...
    ####
    masked_image_features = img_encoder(cam)
    masked_image_features_norm = masked_image_features.norm(dim=-1, keepdim=True)
    masked_image_features_new = masked_image_features / masked_image_features_norm
    new_score = masked_image_features_new[0].dot(target_features_new[0])
    ####

    cam = cam[0].permute(1, 2, 0).data.cpu().numpy()
    cam = np.float32(cam)
    
    plt.imshow(cam)
    
    return new_score


def vit_relevance(image, target_features, img_encoder, device, method="last grad", neg_saliency=False, model_name="ViT-B-32"):
    # 核心函数：计算 ViT 的注意力显著性
    """
        计算 ViT 模型的注意力显著性图

        参数:
        - image: 输入图像张量
        - target_features: 目标特征
        - img_encoder: ViT 图像编码器
        - device: 计算设备
        - method: 可视化方法
        - neg_saliency: 是否生成负显著性图
    """

    # 设置模型为评估模式
    img_encoder.eval()
    # 获取图像特征
    image_features = img_encoder(image) # b32:512; L14:768

    # 文本和图像特征归一化处理
    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features_new = image_features / image_features_norm
    target_features_norm = target_features.norm(dim=-1, keepdim=True)
    target_features_new = target_features / target_features_norm

    # 计算相似度
    similarity = image_features_new[0].dot(target_features_new[0])

    # 根据 neg_saliency 设置目标函数
    if neg_saliency:
        objective = 1-similarity # 负显著性：寻找不相似的区域
    else:
        objective = similarity # 正显著性：寻找相似的区域

    # 反向传播计算梯度
    img_encoder.zero_grad() # 清除之前的梯度
    objective.backward(retain_graph=True) # 计算梯度

    # 获取注意力块和token数量
    # 1. 获取注意力块
    image_attn_blocks = list(dict(img_encoder.transformer.resblocks.named_children()).values())
    # 2. 获取token数量
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]

    # 获取最后一层的注意力权重和梯度
    # 1. 获取主力权重
    last_attn = image_attn_blocks[-1].attn_probs.detach()
    last_attn = last_attn.reshape(-1, last_attn.shape[-1], last_attn.shape[-1])
    # 2. 获取梯度
    last_grad = image_attn_blocks[-1].attn_grad.detach()
    last_grad = last_grad.reshape(-1, last_grad.shape[-1], last_grad.shape[-1])

    # 根据不同方法计算显著性图
    if method=="gradcam":
        # GradCAM方法：使用最后一层的梯度和注意力
        cam = last_grad * last_attn
        cam = cam.clamp(min=0).mean(dim=0) 
        image_relevance = cam[0, 1:] # 去除CLS token
             
    else:
        # 创建单位矩阵作为初始relevance
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        # 遍历所有注意力块
        for blk in image_attn_blocks:
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])

            # 根据方法选择梯度
            if method=="last grad":
                grad = last_grad # 只使用最后一层的梯度
            elif method=="all grads":
                grad = blk.attn_grad.detach() # 使用每一层的梯度
            else:
                print("The available visualization methods are: 'gradcam', 'last grad', 'all grads'.")
                return

            # 计算attention relevance
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0) 
            R += torch.matmul(cam, R) # 累积relevance

        # 提取图像相关的relevance（去除CLS token）
        image_relevance = R[0, 1:]

    # 不同的vision model 得到的image_relavance shape不同
    if model_name=="ViT-B-32":
        image_relevance = image_relevance.reshape(1, 1, 7, 7)
    elif model_name=="ViT-L-14":
        print(image_relevance.shape)
        image_relevance = image_relevance.reshape(1, 1, 16, 16) # 重塑为16x16大小

    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear') # 上采样到224x224

    image_relevance = image_relevance.reshape(224,224).data.cpu().numpy() # 转换为numpy数组
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min()) # 归一化显著性图

    # 处理输入图像
    image = image[0].permute(1, 2, 0).data.cpu().numpy()

    # 归一化图像
    image = (image - image.min()) / (image.max() - image.min())

    
    return image_relevance, image



def interpret_vit(image, target_features, img_encoder, device, method="last grad", neg_saliency=False, width = 224, height=224, model_name = "ViT-B-32"):
    # 主函数：用于解释 Vision Transformer (ViT) 的决策过程
    """
       生成 ViT 模型的注意力可视化图

       参数:
       - image: 输入图像张量
       - target_features: 目标特征
       - img_encoder: ViT 图像编码器
       - device: 计算设备 (CPU/GPU)
       - method: 可视化方法，可选 "last grad", "all grads", "gradcam"
       - neg_saliency: 是否生成负显著性图
   """


    # Func1: Saliency Map Obtaining
    image_relevance, image = vit_relevance(image, target_features, img_encoder, device, method=method, neg_saliency=neg_saliency, model_name=model_name)


    # Func2: Saliency Map Handling
    # 将显著性图叠加到原始图像上
    vis = show_cam_on_image(image, image_relevance, neg_saliency=neg_saliency)

    # 转换图像格式用于显示
    vis = np.uint8(255 * vis) # 转换到0-255范围
    vis = cv2.resize(vis, (width,height))
    # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)# 转换颜色空间

    plt.imshow(vis)
    # plt.imshow(vis)
    # return vis

    
def interpret_vit_overlapped(image, target_features, img_encoder, device, method="last grad"):
    
    pos_image_relevance, _ = vit_relevance(image, target_features, img_encoder, device, method=method, neg_saliency=False)
    neg_image_relevance, image = vit_relevance(image, target_features, img_encoder, device, method=method, neg_saliency=True)

    vis = show_overlapped_cam(image, neg_image_relevance, pos_image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)
    
    
def vit_perword_relevance(image, text, clip_model, clip_tokenizer, device, masked_word="", use_last_grad=True):
    
    clip_model.eval()
    
    main_text = clip_tokenizer(text).to(device)
    # remove the word for which you want to visualize the saliency
    masked_text = re.sub(masked_word, "", text)
    masked_text= clip_tokenizer(masked_text).to(device)
    
    image_features = clip_model.encode_image(image)
    main_text_features = clip_model.encode_text(main_text)
    masked_text_features = clip_model.encode_text(masked_text)
    
    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features_new = image_features / image_features_norm
    main_text_features_norm = main_text_features.norm(dim=-1, keepdim=True)
    main_text_features_new = main_text_features / main_text_features_norm
    
    masked_text_features_norm = masked_text_features.norm(dim=-1, keepdim=True)
    masked_text_features_new = masked_text_features / masked_text_features_norm
  
    objective = image_features_new[0].dot(main_text_features_new[0]-masked_text_features_new[0])
    
    clip_model.visual.zero_grad()
    objective.backward(retain_graph=True)
 
    image_attn_blocks = list(dict(clip_model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    
    last_grad = image_attn_blocks[-1].attn_grad.detach()
    last_grad = last_grad.reshape(-1, last_grad.shape[-1], last_grad.shape[-1])
    
    
    for blk in image_attn_blocks:
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        
        if use_last_grad:
            grad = last_grad
        else:
            grad = blk.attn_grad.detach()
            
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0) 
        R += torch.matmul(cam, R)
        
  
    image_relevance = R[0, 1:]


    #image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = image_relevance.reshape(1, 1, 16, 16)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    
    return image_relevance, image


def interpret_perword_vit(image, text, clip_model, clip_tokenizer, device, masked_word="", use_last_grad=True):
    
    image_relevance, image = vit_perword_relevance(image, text, clip_model, clip_tokenizer, device, masked_word, use_last_grad)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)    
