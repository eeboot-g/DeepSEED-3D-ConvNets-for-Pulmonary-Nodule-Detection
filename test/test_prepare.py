import os

import numpy as np
import SimpleITK as sitk
import pandas
from skimage.morphology import convex_hull_image
from scipy.ndimage import binary_dilation,generate_binary_structure,zoom
from PIL import Image

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

'''
处理mask，
先使用convex_hull_image找到包含整张图像的所有前景的最小凸多边形，并填充成前景，如果凸多边形超过原图1.5倍则采用原图
对前景进行膨胀(3,1)
'''
def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            # 获取最小凸边
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)

    return dilatedMask

'''
截取处理影像窗口，将窗口归一化到[0,255]
'''
def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

# 重新采样
def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

# 世界坐标转体素坐标
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def preprocess_luna():
    islabel = True
    isClean = False
    resolution = np.array([1, 1, 1])
    slice_file = 'data/000.mhd'
    mask_file = 'seg/000.mhd'
    name = '000'
    savepath = 'train'
    luna_label = 'annon/annos.csv'

    sliceim, origin, spacing, isflip = load_itk_image(slice_file)
    if isflip:
        sliceim = sliceim[:, ::-1, ::-1]
    Mask, origin, spacing, isflip = load_itk_image(mask_file)
    if isflip:
        Mask = Mask[:, ::-1, ::-1]
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')

    m1 = Mask == 3
    m2 = Mask == 4
    # 包含左肺和右肺的mask
    Mask = m1 + m2

    xx, yy, zz = np.where(Mask)
    #计算原图的肺部的包围盒
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    #肺部包围盒按重采样后的大小进行缩放
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')

    margin = 5
    # 向左扩展一个margin 向右扩展两个margin
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T

    if isClean:
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        # 膨胀过的前景，包含左右肺（可能为凸多边形）
        dilatedMask = dm1 + dm2
        # 包含左肺和右肺的mask
        Mask = m1 + m2
        # 按位异或，提取出肺轮廓的外轮廓边界作为前景（可能为凸多边形）
        extramask = dilatedMask ^ Mask

        bone_thresh = 210
        pad_value = 170

        #转为 0-255
        sliceim = lumTrans(sliceim)
        #showImg(sliceim[sliceim.shape[0] // 2], False)
        #使用dilatedMask提取出肺部区域，并将影像上其他区域的值设置为170
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype('uint8')
        #showImg(sliceim[sliceim.shape[0] // 2], False)
        # 提取出骨头
        bones = (sliceim*extramask)>bone_thresh
        #showImg(bones[sliceim.shape[0] // 2])
        # 骨头值也设置为170
        sliceim[bones] = pad_value
        #showImg(sliceim[sliceim.shape[0] // 2], False)
        # 重采样
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)

        #使用extendbox对重采样后的影像进行裁剪
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]

        # 转换为4维数组
        sliceim = sliceim2[np.newaxis, ...]
        print(sliceim2.shape)
        print(sliceim.shape)
        #showImg(sliceim2[sliceim2.shape[0] // 2], False)

        # 重采样后按包围框裁剪的影像
        np.save(os.path.join(os.getcwd(), savepath, name + '_clean.npy'), sliceim)
        # 原始spacing
        np.save(os.path.join(os.getcwd(), savepath, name + '_spacing.npy'), spacing)
        # 重采样分辨率下的包围框
        np.save(os.path.join(os.getcwd(), savepath, name + '_extendbox.npy'), extendbox)
        # 原始origin
        np.save(os.path.join(os.getcwd(), savepath, name + '_origin.npy'), origin)
        # 左右肺MASK，原始分辨率下的
        np.save(os.path.join(os.getcwd(), savepath, name + '_mask.npy'), Mask)

    if islabel:
        annos = np.array(pandas.read_csv(luna_label))
        print(annos.shape)
        this_annos = np.copy(annos[annos[:, 0] == float(name)])
        label = []
        if len(this_annos) > 0:
            for c in this_annos:
                print(c)
                #获取标注文件中的标注坐标，计算为体素坐标，标注坐标轴做了调整
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                #生成新的标注信息，将直径的长度变换为体素数量
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            print(label2)
            # 将标注的位置信息按照重采样后的分辨率进行重新计算
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            # 标注的直径也按照重采样的分辨率重新计算
            label2[3] = label2[3] * spacing[1] / resolution[1]
            # 将标注的坐标映射为包围盒内的相对坐标
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            # 转换为标注正常格式
            label2 = label2[:4].T

        np.save(os.path.join(os.getcwd(), savepath, name + '_label.npy'), label2)

def showImg(imgData, Trans=True):
    if Trans:
        img = Image.fromarray(np.array(imgData).astype('int') * 255).convert('L')
    else:
        img = Image.fromarray(np.array(imgData).astype('int')).convert('L')
    img.show()

if __name__=='__main__':
    preprocess_luna()