# -*- coding: utf-8 -*-
"""
PIL库使用
"""

from PIL import Image

# 1. 使用PIL库Image类打开图像, 获取图像属性、缩略图
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    print(im.size, im.format, im.mode)
    im.save('test.png', 'png')
    im.thumbnail((50, 50), resample = Image.BICUBIC)
    im.save('thumbnail.png', 'png')
    im.close()

# 2. 使用PIL库Image类裁剪图像指定区域
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    # 裁剪图像, 从左上角(100, 100)到右下角(600, 600)区域
    box = (100, 100, 600, 600)
    region = im.crop(box)
    region.show()
    
# 3. 使用PIL库Image类对图像进行旋转或者反转
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    # 对原图像逆时针旋转90度, 180度和270度 Image.ROTATE_90, Image.ROTATA_180, Image.ROTATE_270
    im_rotate_90 = im.transpose(Image.ROTATE_90)
    im_rotate_90.show()
    # 对原图像左右翻转、上下翻转 Image.FLIP_LEFT_RIGHT Image.FLIP_TOP_BOTTOM
    im_flip_left_right = im.transpose(Image.FLIP_LEFT_RIGHT)
    im_flip_left_right.show()
    im_flip_top_bottom = im.transpose(Image.FLIP_TOP_BOTTOM)
    im_flip_top_bottom.show()
    
# 4. 使用PIL库将一个图像粘贴到另一个图像
flag = False
if flag:
    im_me = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    im_doraemo = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\doraemo.jpg', 'r')
    im_me.paste(im_doraemo, (0, 150), None)
    im_me.show()
    im_me.paste()
    
# 5. 分离与合并图像颜色通道
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    r, g, b = im.split()
    print(r.show(), g.show(), b.show())
    
    im_merge = Image.merge('RGB', [r, g, b])
    im_merge.show()
    
# 6. 调整图像大小
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    im_resize = im.resize((200, 200))
    im_resize.show()
    im.show()
    im_resize_box = im.resize((864, 648), box = (400, 400, 864, 648))
    im_resize_box.show()
    
# 7. 将图像转换为RGB真彩图、L灰度图、CMYK压缩图
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    im_L = im.convert('L')
    im_L.show()
    im_rgb = im.convert('CMYK')
    im_rgb.show()

# 8. 对原图像进行过滤操作, 包括模糊操作、查找边、角点操作等
if flag:
    from PIL import ImageFilter
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    
    im_blur = im.filter(ImageFilter.BLUR)
    im_blur.show()
    im_find_edges = im.filter(ImageFilter.FIND_EDGES)
    im_find_edges.show()
    # 高斯模糊
    im_gaussian_blur = im.filter(ImageFilter.GaussianBlur)
    im_gaussian_blur.show()
    
# 9. 对原图像中每个像素点进行操作
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    im_point = im.point(lambda x : x / 2)
    im_point.show()
    
# 10. 对原图像进行增强, 包括亮度、对比度等
flag = False
if flag:
    im = Image.open(u'C:\\Users\\Haoran\\Pictures\\Saved Pictures\\me.jpg', 'r')
    from PIL import ImageEnhance
    
    brightness = ImageEnhance.Brightness(im)
    im_brightness = brightness.enhance(1.5)
    im_brightness.show()
    
    contrast = ImageEnhance.Contrast(im)
    im_contrast = contrast.enhance(1.5)
    im_contrast.show()
    
    