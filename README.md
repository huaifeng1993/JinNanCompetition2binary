# 津南挑战赛2二分类模型
x光机限制品非限制品二分类。
本代码为基于inception_V3的normal 和restricted 物品分类模型。
物体检测模型链接为 [津南2物体检测](https://github.com/huaifeng1993/JinNanCompetition2). 
## 目录结构
```
|--root
    |--*.py
    |--logs
    |--*.ipynb
```

* main.py 为训练的主文件代码。
* data.py为数据类。
* model.py为使用的模型结构。
* generator.py 为数据生成器。
* config.py 保存配置参数。
* test.py 生成而分类结果。
* result_merge.ipynb 合成检测模型和二分类模型的结果，目的为滤掉，物体检测模型的误检的图片结果。
## 训练与测试
在main.py和test.py中注意更改数据集路径。
* 训练 python main.py
* 测试 python test.py

## 合成结果
需要把物体检测模型的生成的json结果文件复制到本根目录，之后一路shift+enter。