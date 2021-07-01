# 应用SANET和单目深度的任意风格迁移模型

本代码复现了参考文献[1]的模型

参考文献:
[1]`Arbitrary Style Transfer with Style-Attentional Networks`

---

## 如何训练

打开**train.py**文件，修改文件中的模型位置，训练集位置，参数，即可开始训练模型。

```
parser.add_argument('--content_dir', type=str, default='./train2014',
help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='./train',
help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth'
parser.add_argument('--save_dir', default='./experiments',
help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=1000)
parser.add_argument('--start_iter', type=float, default=0)
```

---

## 如何测试

**对一张图片进行风格迁移**：打开**eval.py**文件，修改parser中的模型路径，在main函数中修改要转风格的内容图片路径和风格图片路径，运行即可

**利用电脑摄像头实时风格迁移**：直接运行**video.py**文件即可

**对一段视频进行风格迁移**：打开**video.py**文件，在主函数调用tranfrom_video函数即可，需要输入视频路径和风格图片路径

---

## 关于各文件夹和文件的注解
**experiments**：用来保存训练过程的编码器和解码器模型

**input**：用来存储测试时需要风格迁移的图片

**output**：用来存储测试时风格迁移后的图片

**style**：用来存储测试时的风格图片

**train**：用来存储训练时的风格图片

**train_2014**：用来存储训练时没有风格的图片

**Pytorch**：存有提取图片深度图的模型

**transformer.pth**：预训练的编码器

**decoder.pth**：预训练的解码器

**optimizer.pth**：预训练的优化器

**torch_model.pth**：预训练的深度提取网络

**vgg_normalised.pth**：预训练的VGG19

**depth_info.py**：用来提取图片深度

**Train.py**：用来训练风格迁移网络

**Eval.py**：用来测试风格迁移网络

**video.py**：用来对视频进行风格迁移和实时风格迁移

