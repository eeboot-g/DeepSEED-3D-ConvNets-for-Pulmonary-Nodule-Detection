import os
import shutil
import numpy as np
from config_training import config

from scipy.io import loadmat
import numpy as np
# import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
import warnings
import pdb

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

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

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


def savenpy_luna(id, annos, filelist, luna_segment, luna_data,savepath):
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
#     resolution = np.array([2,2,2])
    name = filelist[id]
    
    sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data,name+'.mhd'))
    Mask,origin,spacing,isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        sliceim = sliceim[:, ::-1, ::-1]
        Mask = Mask[:,::-1,::-1]
        print('flip!')

    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    # 生成二值图
    m1 = Mask==3
    m2 = Mask==4
    # 包含左肺和右肺的mask
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    # 计算原图的肺部的包围盒
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    # 肺部包围盒按重采样后的大小进行缩放
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    # 向左扩展一个margin 向右扩展两个margin
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
    
    this_annos = np.copy(annos[annos[:,0]==float(name)])

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        # 膨胀过的前景，包含左右肺（可能为凸多边形）
        dilatedMask = dm1+dm2
        # 包含左肺和右肺的mask
        Mask = m1+m2
        # 按位异或，提取出肺轮廓的外轮廓边界作为前景（可能为凸多边形）
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        # CT值转为 0-255
        sliceim = lumTrans(sliceim)
        # 使用dilatedMask提取出肺部区域，并将影像上其他区域的值设置为170
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        # 提取出骨头
        bones = (sliceim*extramask)>bone_thresh
        # 骨头值也设置为170
        sliceim[bones] = pad_value

        # 安装指定的体素进行重采样
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)

        # 使用extendbox对重采样后的影像进行裁剪
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]

        # 转换为4维数组
        sliceim = sliceim2[np.newaxis,...]
        # 重采样后按包围框裁剪的影像
        np.save(os.path.join(savepath, name+'_clean.npy'), sliceim)
        # 原始spacing
        np.save(os.path.join(savepath, name+'_spacing.npy'), spacing)
        # 重采样分辨率下的包围框
        np.save(os.path.join(savepath, name+'_extendbox.npy'), extendbox)
        # 原始origin
        np.save(os.path.join(savepath, name+'_origin.npy'), origin)
        # 左右肺MASK，原始分辨率下的
        np.save(os.path.join(savepath, name+'_mask.npy'), Mask)

    if islabel:
        this_annos = np.copy(annos[annos[:,0]==float(name)])
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                # 获取标注文件中的标注坐标，计算为体素坐标，标注坐标轴做了调整
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                # 生成新的标注信息，将直径的长度变换为体素数量
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath,name+'_label.npy'), label2)
        
    print(name)


def preprocess_luna0():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd') ]
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        
        pool = Pool()
        partial_savenpy_luna = partial(savenpy_luna,annos=annos,filelist=filelist,
                                       luna_segment=luna_segment,luna_data=luna_data,savepath=savepath)

        N = len(filelist)
        #savenpy(1)
        _=pool.map(partial_savenpy_luna,range(N))
        pool.close()
        pool.join()
    print('end preprocessing luna')
    f= open(finished_flag,"w+")


def preprocess_luna():
    luna_segment = config['luna_segment']
    savepath = config['preprocess_result_path']
    luna_data = config['luna_data']
    luna_label = config['luna_label']
    finished_flag = '.flag_preprocessluna'
    print('starting preprocessing luna')
    if not os.path.exists(finished_flag):
        filelist = [f.split('.mhd')[0] for f in os.listdir(luna_data) if f.endswith('.mhd')]
        annos = np.array(pandas.read_csv(luna_label))

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        pool = Pool(15)
        partial_savenpy_luna = partial(savenpy_luna, annos=annos, filelist=filelist,
                                       luna_segment=luna_segment, luna_data=luna_data, savepath=savepath)

        for i in range(len(filelist)):
            pool.apply_async(partial_savenpy_luna, args=(i,))

        pool.close()
        pool.join()
    print('end preprocessing luna')
    f = open(finished_flag, "w+")
    
def prepare_luna():
    print('start changing luna name')
    luna_raw = config['luna_raw']
    luna_abbr = config['luna_abbr']
    luna_data = config['luna_data']
    luna_segment = config['luna_segment']
    finished_flag = '.flag_prepareluna'
    
    if not os.path.exists(finished_flag):

        subsetdirs = [os.path.join(luna_raw,f) for f in os.listdir(luna_raw) if f.startswith('subset') and os.path.isdir(os.path.join(luna_raw,f))]
        if not os.path.exists(luna_data):
            os.mkdir(luna_data)

#         allnames = []
#         for d in subsetdirs:
#             files = os.listdir(d)
#             names = [f[:-4] for f in files if f.endswith('mhd')]
#             allnames = allnames + names
#         allnames = np.array(allnames)
#         allnames = np.sort(allnames)

#         ids = np.arange(len(allnames)).astype('str')
#         ids = np.array(['0'*(3-len(n))+n for n in ids])
#         pds = pandas.DataFrame(np.array([ids,allnames]).T)
#         namelist = list(allnames)
        
        abbrevs = np.array(pandas.read_csv(config['luna_abbr'],header=None))
        namelist = list(abbrevs[:,1])
        ids = abbrevs[:,0]
        
        for d in subsetdirs:
            files = os.listdir(d)
            files.sort()
            for f in files:
                name = f[:-4]
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)
                shutil.move(os.path.join(d,f),os.path.join(luna_data,filename+f[-4:]))
                print(os.path.join(luna_data,str(id)+f[-4:]))

        files = [f for f in os.listdir(luna_data) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_data,file),'r') as f:
                content = f.readlines()
                id = file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.raw\n'
                print(content[-1])
            with open(os.path.join(luna_data,file),'w') as f:
                f.writelines(content)

                
        seglist = os.listdir(luna_segment)
        for f in seglist:
            if f.endswith('.mhd'):

                name = f[:-4]
                lastfix = f[-4:]
            else:
                name = f[:-5]
                lastfix = f[-5:]
            if name in namelist:
                id = ids[namelist.index(name)]
                filename = '0'*(3-len(str(id)))+str(id)

                shutil.move(os.path.join(luna_segment,f),os.path.join(luna_segment,filename+lastfix))
                print(os.path.join(luna_segment,filename+lastfix))


        files = [f for f in os.listdir(luna_segment) if f.endswith('mhd')]
        for file in files:
            with open(os.path.join(luna_segment,file),'r') as f:
                content = f.readlines()
                id =  file.split('.mhd')[0]
                filename = '0'*(3-len(str(id)))+str(id)
                content[-1]='ElementDataFile = '+filename+'.zraw\n'
                print(content[-1])
            with open(os.path.join(luna_segment,file),'w') as f:
                f.writelines(content)
    print('end changing luna name')
    f= open(finished_flag,"w+")
    
if __name__=='__main__':
    prepare_luna()
    preprocess_luna()






