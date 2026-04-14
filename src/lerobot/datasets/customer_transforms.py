from PIL import Image
import torchvision.transforms as transforms
from typing import Any

class CustomerImageTransforms:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        img_transform_list = []
        # 随机缩放（0.7~1.3 倍），再裁剪为 256x256
        if "random_resize_crop" in self._cfg:
            random_scale_crop = self._cfg["random_resize_crop"]
            self.random_scale_crop = transforms.RandomResizedCrop(
                size=random_scale_crop["size"],    # 最终输出尺寸 h w
                scale=random_scale_crop["scale"],   # 缩放范围：最小 0.7 倍（缩小），最大 1.3 倍（放大）
                ratio=random_scale_crop["ratio"],   # 固定长宽比为 1:1（正方形）
                interpolation=random_scale_crop["interpolation"]  # 高质量插值（适合缩小图像）
            )
            img_transform_list.append(self.random_scale_crop)
        # 随机仿射变换--平移
        if "randorm_affine" in self._cfg:
            random_affine = self._cfg["randorm_affine"]
            self.random_affine = transforms.RandomAffine(
                degrees=random_affine['degrees'],
                translate=random_affine['translate'],
                scale=random_affine['scale']
            )
            img_transform_list.append(self.random_affine)
        
        # 随机旋转 ±15 度，空白区域用白色填充，旋转后扩展图像
        if "random_rotation" in self._cfg:
            random_rotate = self._cfg["random_rotation"]
            self.random_rotate = transforms.RandomRotation(
                degrees=random_rotate["degrees"],  # 角度范围：-15° 到 15°
                expand=random_rotate["expand"],        # 扩展图像，避免裁剪
                fill=random_rotate["fill"],  # 白色填充
                # resample=Image.BILINEAR  # 新版 torchvision 0.12.x以上 双线性重采样（画质更平滑）
                interpolation=random_rotate["interpolation"]  # 旧版本用 PIL.Image 的常量（而非字符串）
            )
            img_transform_list.append(self.random_rotate)
        # 随机调整亮度、对比度、饱和度、色相
        if "colorjitter" in self._cfg:
            random_color_jitter = self._cfg["colorjitter"]
            self.random_color_jitter = transforms.ColorJitter(
                brightness=random_color_jitter["brightness"],  # 亮度 ±20%
                contrast=random_color_jitter["contrast"],    # 对比度 ±20%
                saturation=random_color_jitter["saturation"],  # 饱和度 ±20%
                hue=random_color_jitter["hue"]          # 色相 ±0.1（避免颜色失真）
            )
            img_transform_list.append(self.random_color_jitter)
        
        self.train_img_transform = transforms.Compose(img_transform_list)

        self.random_erase = None
        if "randomerase" in self._cfg:
             # p=0.5：50%概率执行；scale=(0.02, 0.33)：擦除区域占比2%~33%
             self.random_erase = transforms.RandomErasing(
                 p=self._cfg["randomerase"]["p"], 
                 scale=self._cfg["randomerase"]["scale"], 
                 ratio=self._cfg["randomerase"]["ratio"], 
                 value=self._cfg["randomerase"]["value"]
                 )
    
    def __call__(self, *inputs: Any) -> Any:
        return self.train_img_transform(*inputs)

